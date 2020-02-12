from utils.data_loader_kine_mus import *

import tensorflow as tf
from tensorflow.python.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import os


class CyclicalAnnealingSchedule(tf.keras.callbacks.Callback):

    def __init__(self, cycles, database_size, epochs, log_dir=None):
        training_instances = database_size * epochs
        self.cycle_duration = training_instances / cycles
        self.log_dir = log_dir
        self.processed_instances = 0
        self.instances_offset = 0
        self.absolute_batch = 0
        self.batch_offset = 0

    def on_train_begin(self, logs=None):
        if self.model.kl_loss_weight is None:
            raise Exception("VAE model must have a kl_loss_weight [`tf.Variable`] attribute")
        self.model.kl_loss_weight.assign(0)

    def on_train_batch_end(self, batch, logs=None):
        self.processed_instances += logs['size']
        self.absolute_batch = batch + self.batch_offset
        state = (self.processed_instances - self.instances_offset) % self.cycle_duration

        if state <= self.cycle_duration / 3:
            self.model.kl_loss_weight.assign(0)

        elif self.cycle_duration/3 <= state <= 2*self.cycle_duration/3:
            self.model.kl_loss_weight.assign((3 / self.cycle_duration) * (state - self.cycle_duration/3))

        elif self.cycle_duration >= state >= 2*self.cycle_duration/3:
            self.model.kl_loss_weight.assign(1)

        elif state > self.cycle_duration:  # Restart cycle
            self.instances_offset = self.processed_instances

        if self.log_dir:
            with tf.summary.create_file_writer(os.path.join(self.log_dir, 'train')).as_default():
                tf.summary.scalar('kl_loss_weight', self.model.kl_loss_weight, step=self.absolute_batch)

    def on_epoch_end(self, epoch, logs=None):
        self.batch_offset = self.absolute_batch


class EmbeddingSpaceLogger(tf.keras.callbacks.Callback):

    def __init__(self, df, X, log_dir, name='train'):
        self.log_dir = log_dir
        self.df = df
        self.X = X
        self.name = name

    def on_train_end(self, logs=None):
        print("Saving embeddings to tf projector in %s" % self.log_dir)
        z_mean, z_log_var, z = self.model.encoder(self.X)
        z_mean = np.array(z_mean)

        # Save embeddings for each data instance
        data_df = pd.DataFrame(data=z_mean)
        data_df.to_csv(self.log_dir + "/vecs.tsv", sep='\t', index=False, header=False)

        # Save metadata tsv file
        metadata = self.df[[e.value for e in ExperimentFields]]
        metadata.to_csv(self.log_dir + "/meta.tsv", sep='\t', index=False)

    def on_train_begin(self, logs=None):
        # Save an embedding projection before training
        self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % 2 != 0:
            return
        print("printing embedding")
        phases = self.df[ExperimentFields.phase.value].values

        figure = plt.figure(figsize=(10, 10))
        z_mean, z_log_var, z = self.model.encoder(self.X)
        print(z_mean.shape)
        if z_mean.shape[1] > 2:
            return
        print(phases.shape)
        z_mean = np.array(z_mean)
        plt.scatter(z_mean[phases == 2, 0], z_mean[phases == 2, 1], c='r', alpha=1, label="Manipulation")
        plt.scatter(z_mean[phases == 3, 0], z_mean[phases == 3, 1], marker='+', c='k', alpha=0.5, label="Retreat")
        plt.scatter(z_mean[phases == 1, 0], z_mean[phases == 1, 1], marker='2', c='b', alpha=0.5, label="Approach")

        plt.legend()
        #
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        with tf.summary.create_file_writer(os.path.join(self.log_dir, 'train')).as_default():
            tf.summary.image("Embedding Space", image, step=epoch)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, using the reparametrization trick to allow back propagation."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps Kinematic Joint Positions to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
                 latent_dim=32,
                 intermediate_dim=64,
                 name='Encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu', name="encoder_hidden_layer")
        self.dense_mean = layers.Dense(latent_dim, name="mean_layer")
        self.dense_log_var = layers.Dense(latent_dim, name="log_var_layer")
        self.sampling = Sampling(name="Gaussian_Sampling")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
                 original_dim,
                 intermediate_dim=64,
                 name='Decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu', name="decoder_hidden_layer")
        # Output layer without activation to reproduce kinematic angle data
        self.dense_output = layers.Dense(original_dim, name="output_layer")

    def call(self, inputs):
        # Inputs are the
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class KLDivergence(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, initial_weight=0.0,
                 name='KL-Divergence',
                 **kwargs):
        super(KLDivergence, self).__init__(name=name, **kwargs)

        self.kl_loss_weight = tf.Variable(initial_weight, trainable=False, name="kl_loss_weight", dtype=tf.float32)

    def call(self, inputs):
        z_mean, z_log_var, z = inputs
        # Add KL divergence regularization loss for this forward pass
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        return self.kl_loss_weight * kl_loss

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kl_loss_weight': self.kl_loss_weight,
        })
        return config


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim=18,
                 intermediate_dim=64,
                 latent_dim=2,
                 kl_loss_weight=0.2,
                 kl_loss_metric_name='kl_loss',
                 name='Autoencoder',
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                               intermediate_dim=intermediate_dim)

        self.kl_loss_weight = tf.Variable(kl_loss_weight, dtype=tf.float32, trainable=False, name='kl_loss_weight')

        self.kl_loss_metric_name = kl_loss_metric_name

    def call(self, inputs):
        # Forward pass of the Encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        # Forward pass of the Decoder taken the re-parameterized z latent variable
        reconstructed = self.decoder(z)

        # Compute KL loss term (and add it as a metric)
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_metric(kl_loss, aggregation='mean', name=self.kl_loss_metric_name)

        # Weight KL loss with the provided `betha` weight parameter
        weighted_kl_loss = self.kl_loss_weight * kl_loss
        # Add weighted KL term as loss and metric
        self.add_loss(weighted_kl_loss)
        self.add_metric(weighted_kl_loss, aggregation='mean', name="weighted_" + self.kl_loss_metric_name)

        return reconstructed

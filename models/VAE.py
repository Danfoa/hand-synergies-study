import tensorflow as tf
from tensorflow.python.keras import layers


class KLLossLogger(tf.keras.callbacks.Callback):

    def __init__(self, train_summary_writer, batch_size=1):
        self.train_summary_writer = train_summary_writer
        self.batch_size = batch_size
        pass

    def on_train_batch_end(self, batch, logs=None):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('kl_loss_weight', self.model.kl_loss_weight, step=batch*self.batch_size)
            # tf.summary.scalar('kl_loss', tf.keras.metrics.Mean(self.model.kl_loss).result(), step=batch*self.batch_size)
    # def on_train_epoch_end(self, epocj, logs=None):
        # print("Epoch: KL-Loss %.5f" % )

class CyclicalAnnealingSchedule(tf.keras.callbacks.Callback):

    def __init__(self, kl_decay_rate, batch_size):
        self.kl_decay_rate = kl_decay_rate
        self.batch_size = batch_size
        pass

    def on_train_batch_end(self, batch, logs=None):
        # print('For batch {}, kl_weight: {}'.format(batch, self.model.kl_loss_weight))
        self.model.kl_loss_weight = 1 / (1 + (1 - 0.0001) / 0.0001 * tf.exp(-self.kl_decay_rate * batch * self.batch_size))
        # tf.summary.scalar('batch_kl_loss_weight', self.model.kl_loss_weight, step=batch * self.batch_size)


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
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
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
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        # Output layer without activation to reproduce kinematic angle data
        self.dense_output = layers.Dense(original_dim)

    def call(self, inputs):
        # Inputs are the
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim=18,
                 intermediate_dim=64,
                 latent_dim=2,
                 name='Autoencoder',
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                               intermediate_dim=intermediate_dim)
        self.kl_loss_weight = 0
        self.kl_loss = 0

    def call(self, inputs):
        # Forward pass of the Encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        # Forward pass of the Decoder taken the re-parameterized z latent variable
        reconstructed = self.decoder(z)

        # Add KL divergence regularization loss for this forward pass
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

        self.add_loss(self.kl_loss_weight * kl_loss)
        self.kl_loss = kl_loss
        return reconstructed

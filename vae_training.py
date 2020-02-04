from utils.data_loader_kine_mus import *

from models.VAE import VariationalAutoEncoder, CyclicalAnnealingSchedule, Encoder, Decoder, KLDivergence
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io


class ImageLogger(tf.keras.callbacks.Callback):

    def __init__(self, df, X, train_summary_writer, name=''):
        self.train_summary_writer = train_summary_writer
        self.df = df
        self.X = X
        self.name = name

    def on_train_begin(self, logs=None):
        self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs=None):
        print("printing embedding")
        phases = self.df[ExperimentFields.phase.value].values

        figure = plt.figure(figsize=(10, 10))
        z_mean, z_log_var, z = self.model.layers[1](self.X)
        print(z_mean.shape)
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

        with self.train_summary_writer.as_default():
            tf.summary.image("Embedding Space", image, step=epoch)


if __name__ == '__main__':
    # Load the dataset
    df = load_subjects_data(DATABASE_PATH, subjects_id=SUBJECTS)

    # Shuffle data for training
    # Split the data into training and testing
    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=True)
    del df

    X_train = df_train[[e.value for e in RightHand]].values
    X_train = StandardScaler(copy=False).fit_transform(X_train)
    X_test = df_test[[e.value for e in RightHand]].values
    X_test = StandardScaler(copy=False).fit_transform(X_test)

    train_phase = df_train[ExperimentFields.phase.value]
    test_phase = df_test[ExperimentFields.phase.value]

    num_instances = X_train.shape[0]

    HP_LR = hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.01))
    HP_LATENT_DIM = hp.HParam('latent_dim', hp.Discrete([2, 3, 4, 5]))
    HP_HD = hp.HParam('hidden_dimensions', hp.Discrete([30, 50, 80, 100]))
    HP_BATCH_SIZE = hp.HParam('hidden_dimensions', hp.Discrete([2, 12, 32]))

    batch_size = 64
    intermediate_dim = 100
    lr = 1.12e-3

    for batch_size in [32, 16]:
        for intermediate_dim in [75, 30, 90, 150]:
            for lr in np.linspace(0.011, 0.0001, 10):
                latent_dim = 2
                epochs = 15
                # Callbacks
                log_dir = "models/vae-logs/lr=%.5f-hd=%d-lat_dim=%d-b=%d/" % (lr, intermediate_dim, latent_dim,
                                                                              batch_size)
                train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')

                # _______________________________________________________________________________
                # vae = VariationalAutoEncoder(original_dim=18, intermediate_dim=intermediate_dim, latent_dim=latent_dim)

                original_dim = 18
                input = tf.keras.layers.Input(shape=(original_dim,))

                encoder = Encoder(latent_dim=latent_dim,
                                  intermediate_dim=intermediate_dim)
                decoder = Decoder(original_dim=original_dim,
                                  intermediate_dim=intermediate_dim)
                kl_divergence = KLDivergence(initial_weight=0)
                # Forward pass of the Encoder
                z_mean, z_log_var, z = encoder(input)
                # Forward pass of the Decoder taken the re-parameterized z latent variable
                reconstructed = decoder(z)

                kl_loss = kl_divergence((z_mean, z_log_var, z))

                vae = tf.keras.models.Model(input, reconstructed, name='Autoencoder')
                vae.add_loss(kl_loss)

                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

                vae.compile(optimizer,
                            loss=tf.keras.losses.MeanSquaredError(),
                            metrics=[tf.keras.metrics.MeanAbsoluteError(name="reconstruction_MAE_loss"),
                                     tf.keras.metrics.MeanSquaredError(name="reconstruction_MSE_loss")])

                # Add visualization of the KL_loss
                vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')

                # _______________________________________________________________________________

                print(vae.layers)

                tf.keras.utils.plot_model(vae, 'vae_model.png', show_shapes=True)

                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch',
                                                                      profile_batch=0)

                vae.fit(X_train, X_train,
                        epochs=epochs,
                        shuffle=True,
                        batch_size=batch_size,
                        validation_split=0.0,
                        workers=4,
                        callbacks=[
                                   CyclicalAnnealingSchedule(cycle_duration=1.87e6,
                                                             summary_writer=train_summary_writer),
                                   tensorboard_callback,
                                   tf.keras.callbacks.TerminateOnNaN(),
                                   ImageLogger(df_train, X_train, train_summary_writer)
                        ])

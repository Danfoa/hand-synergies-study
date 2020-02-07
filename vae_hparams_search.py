from utils.data_loader_kine_mus import *

from models.VAE import VariationalAutoEncoder, EmbeddingSpaceLogger, CyclicalAnnealingSchedule, Encoder, Decoder, KLDivergence
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_compiled_vae_model(latent_dim, intermediate_dim, lr=0.001):
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
    return vae


if __name__ == '__main__':
    # Load the dataset
    df = load_subjects_data(DATABASE_PATH, subjects_id=SUBJECTS)

    # Shuffle data for training
    # Split the data into training and testing
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
    del df

    X_train = df_train[[e.value for e in RightHand]].values
    X_train = StandardScaler(copy=False).fit_transform(X_train)
    X_test = df_test[[e.value for e in RightHand]].values
    X_test = StandardScaler(copy=False).fit_transform(X_test)

    num_instances = X_train.shape[0]

    HP_LR = hp.HParam('learning_rate', hp.RealInterval(0.00001, 0.001))
    HP_LATENT_DIM = hp.HParam('latent_dim', hp.Discrete([2, 3, 4, 5]))
    HP_HD = hp.HParam('hidden_dimensions', hp.Discrete([30, 50, 80, 100, 150, 300]))
    HP_ANNEALING_CYCLES = hp.HParam('annealing_cycles', hp.Discrete([1, 2, 4, 6, 8]))

    hparams_log_dir = "models/vae-logs"
    hparams_writer = tf.summary.create_file_writer(hparams_log_dir)

    TEST_RECONSTRUCTION = 'test_reconstruction_MSE'
    with hparams_writer.as_default():
        hp.hparams_config(
            hparams=[HP_LR, HP_HD, HP_LATENT_DIM, HP_ANNEALING_CYCLES],
            metrics=[hp.Metric('epoch_loss', display_name='epoch_loss'),
                     hp.Metric('epoch_kl_loss', display_name='epoch_kl_loss'),
                     hp.Metric('epoch_reconstruction_MAE_loss', display_name='epoch_MAE_loss')]
        )

    epochs = 2
    batch_size = 100
    for latent_dim in HP_LATENT_DIM.domain.values:
        for intermediate_dim in [75, 30, 90, 150]:
            for lr in np.linspace(HP_LR.domain.min_value, HP_LR.domain.max_value, 10):
                for cycles in HP_ANNEALING_CYCLES.domain.values:
                    hparams = {
                        HP_LATENT_DIM: latent_dim,
                        HP_HD: intermediate_dim,
                        HP_LR: lr,
                        HP_ANNEALING_CYCLES: cycles,
                    }
                    print("\n\nTraining model with: \n", hparams)
                    # Create individual log files to save embeddings
                    log_dir = "models/vae-logs/lr=%.5f-hd=%d-lat_dim=%d-cycles=%d" % (lr, intermediate_dim,
                                                                                       latent_dim, cycles)

                    vae = get_compiled_vae_model(lr=lr, latent_dim=latent_dim, intermediate_dim=intermediate_dim)
                    print(vae.layers)

                    vae.fit(X_train, X_train,
                            epochs=epochs,
                            shuffle=True,
                            batch_size=batch_size,
                            validation_split=0.0,
                            workers=4,
                            callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                       tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                                      update_freq='batch', profile_batch=0),
                                       hp.KerasCallback(hparams_log_dir, hparams),
                                       CyclicalAnnealingSchedule(cycles=cycles, database_size=X_train.shape[0],
                                                                 epochs=epochs, log_dir=log_dir),
                                       EmbeddingSpaceLogger(df_train, X_train, log_dir)
                                       ])
                    # Evaluate model with test set
                    X_reconstructed = vae.predict(X_test)

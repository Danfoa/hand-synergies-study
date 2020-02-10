from utils.data_loader_kine_mus import *

from models.VAE import VariationalAutoEncoder, EmbeddingSpaceLogger, CyclicalAnnealingSchedule, Encoder, Decoder
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorboard.plugins.hparams import api as hp

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    HP_ANNEALING_CYCLES = hp.HParam('annealing_cycles', hp.Discrete([0, 1, 2, 4, 6, 8]))

    hparams_log_dir = "models/vae-logs/fuck"
    hparams_writer = tf.summary.create_file_writer(hparams_log_dir)

    TEST_RECONSTRUCTION = 'test_reconstruction_MSE'
    with hparams_writer.as_default():
        hp.hparams_config(
            hparams=[HP_LR, HP_HD, HP_LATENT_DIM, HP_ANNEALING_CYCLES],
            metrics=[
                     # hp.Metric('batch_MSE_loss', group="train", display_name='batch_MSE_loss'),
                     # hp.Metric('epoch_MSE_loss', group="train", display_name='epoch_MSE_loss'),
                     # hp.Metric('batch_kl_loss', group="train", display_name='batch_kl_loss'),
                     hp.Metric('epoch_kl_loss', display_name='epoch_kl_loss'),
                     hp.Metric('batch_loss', display_name='batch_loss'),
                     hp.Metric('epoch_loss', display_name='epoch_loss'),
                     ]
        )

    epochs = 2
    batch_size = 100
    for latent_dim in HP_LATENT_DIM.domain.values:
        for intermediate_dim in HP_HD.domain.values:
            for lr in np.linspace(HP_LR.domain.max_value, HP_LR.domain.min_value, 10):
                for cycles in HP_ANNEALING_CYCLES.domain.values:
                    hparams = {
                        HP_LATENT_DIM: latent_dim,
                        HP_HD: intermediate_dim,
                        HP_LR: lr,
                        HP_ANNEALING_CYCLES: cycles,
                    }
                    # print("\n\nTraining model 
                    # print([param.name])
                    # Create individual log files to save embeddings
                    log_dir = "models/vae-logs/fuck/lr=%.5f-hd=%d-lat_dim=%d-cycles=%d" % (lr, intermediate_dim,
                                                                                      latent_dim, cycles)

                    # vae = get_compiled_vae_model(lr=lr, latent_dim=latent_dim, intermediate_dim=intermediate_dim)
                    vae = VariationalAutoEncoder(original_dim=18,
                                                 latent_dim=latent_dim,
                                                 intermediate_dim=intermediate_dim,
                                                 kl_loss_weight=0.2)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    vae.compile(optimizer,
                                loss=tf.keras.losses.MeanSquaredError(),
                                metrics=[
                                         # tf.keras.metrics.MeanAbsoluteError(name="reconstruction_MAE_loss"),
                                         tf.keras.metrics.MeanSquaredError(name="MSE_loss")])

                    # Define callback to store embeddings at each epoch
                    vae_callbacks = [
                        # EmbeddingSpaceLogger(df_test, X_test, log_dir)
                    ]
                    # If cyclical annealing scheduling is desired, define it
                    if cycles > 0:
                        vae_callbacks.append(CyclicalAnnealingSchedule(cycles=cycles, database_size=X_train.shape[0],
                                                                       epochs=epochs))

                    vae.fit(X_train, X_train,
                            epochs=epochs,
                            shuffle=False,
                            batch_size=batch_size,
                            validation_split=0.0,
                            workers=4,
                            callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                       tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                                      update_freq='batch', profile_batch=0),
                                       hp.KerasCallback(log_dir + '/train', hparams, trial_id=log_dir)
                                       ] + vae_callbacks)

                    # Evaluate model with test set
                    X_reconstructed = vae.predict(X_test)



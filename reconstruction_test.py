
from models.VAE import VariationalAutoEncoder, EmbeddingSpaceLogger, CyclicalAnnealingSchedule, Encoder, Decoder
import tensorflow as tf
import os
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorboard.plugins.hparams import api as hp

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.patheffects as mpe

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

from utils.data_loader_kine_mus import *

from utils.visualization import plot_reconstruction_error


if __name__ == '__main__':
    # Load the dataset
    df = load_subjects_data(DATABASE_PATH, subjects_id=SUBJECTS)
    df.fillna(value=0, inplace=True)

    # ADLs to monitor
    traj = []
    for id in [1, 26, 14, 4]: #[6, 9, 1, 10, 19]:
        traj.append(df[df[ExperimentFields.adl_ids.value] == id].copy())

    print(df.shape)

    joint_names = [e.value for e in RightHand]
    models_labels = []
    models_mae_errors = []

    # Shuffle data for training
    # Split the data into training and testing
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
    del df

    train_scaler = StandardScaler()
    X_train = df_train[[e.value for e in RightHand]].values
    X_train_scaled = train_scaler.fit_transform(X_train)

    test_scaler = StandardScaler()
    X_test = df_test[[e.value for e in RightHand]].values
    X_test_scaled = test_scaler.fit_transform(X_test)

    for t in traj:
        t[joint_names] = train_scaler.transform(t[joint_names].values)

    for latent_dim in [2]:  #3, 4, 5, 6]:
        original_dim = X_train_scaled.shape[1]

        # Test PCA reconstruction ____________________________________________________
        pca = PCA(n_components=latent_dim, svd_solver='full')

        pca.fit(X_train)
        X_test_transformed = pca.transform(X_test)

        # X_train_reconstructed = pca.inverse_transform(X_train_transformed)
        X_test_reconstructed = pca.inverse_transform(X_test_transformed)

        mae_pca = pd.DataFrame(np.abs(X_test - X_test_reconstructed), columns=joint_names)
        models_mae_errors.append(mae_pca)
        models_labels.append("PCA - MAE:%.3f" % tf.math.reduce_mean(tf.keras.losses.MAE(X_test, X_test_reconstructed)))

        log_dir = "models/vae-logs/visual-embeddings"

        # Test VAE reconstruction ____________________________________________________
        for intermediate_dim in [30]:
            for kl_max_weight in np.linspace(0.03, 0.001, 7):
                for cycles in [1, 2, 4]:
                    lr = 0.001
                    # Test VAE enconding reconstruction
                    vae = VariationalAutoEncoder(original_dim=original_dim,
                                                 latent_dim=latent_dim,
                                                 intermediate_dim=intermediate_dim,
                                                 kl_loss_weight=0.0)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    vae.compile(optimizer,
                                loss=tf.keras.losses.MeanAbsoluteError(),
                                metrics=[
                                        # tf.keras.metrics.MeanAbsoluteError(name="MAE_loss"),
                                        tf.keras.metrics.MeanSquaredError(name="MSE_loss")])

                    run_id = "lr=%.5f-hd=%d-lat_dim=%d-kl_w=%.3f_%d-cycles" % (lr, intermediate_dim,
                                                                     latent_dim, kl_max_weight, cycles)
                    run_log_dir = os.path.join(log_dir, run_id)

                    batch_size = 32
                    epochs = 10
                    vae.fit(X_train_scaled, X_train_scaled,
                            epochs=epochs,
                            shuffle=False,
                            batch_size=batch_size,
                            validation_split=0.0,
                            workers=4,
                            callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                       tf.keras.callbacks.TensorBoard(log_dir=run_log_dir, histogram_freq=5,
                                                                      update_freq='batch', profile_batch=0),
                                       CyclicalAnnealingSchedule(cycles=cycles, max_weight=kl_max_weight,
                                                                 database_size=X_train_scaled.shape[0],
                                                                 epochs=epochs),
                                       EmbeddingSpaceLogger(df_test, X_test_scaled, run_log_dir, traj=traj),
                                       ]
                            )

                    X_train_reconstructed_vae = vae.predict(X_train)
                    vae.save_weights("models/VAE-ld=%d/VAE" % latent_dim, save_format="tf")
                    X_test_reconstructed_vae = train_scaler.inverse_transform(vae.predict(X_test_scaled))

                    mae_vae = pd.DataFrame(np.abs(X_test - X_test_reconstructed_vae), columns=joint_names)
                    mae = tf.math.reduce_mean(tf.keras.losses.MAE(X_test, X_test_reconstructed_vae))
                    models_mae_errors.append(mae_vae)
                    models_labels.append("VAE[hd:%d][kl_w:%.1e] - MAE:%.3f" % (intermediate_dim, kl_max_weight, mae))

                    error_vector_test = mae_vae.mean(axis=0).values
                    print("Test VAE")
                    print(error_vector_test)
                    print("MAE %.5f" % mae)

        error_vector_test = mae_pca.mean(axis=0).values
        print("Test PCA")
        print(error_vector_test)
        print("MAE %.5f" % tf.math.reduce_mean(tf.keras.losses.MAE(X_test, X_test_reconstructed)))

        title = "Reconstruction MAE errors [LatDim:%d]" % latent_dim
        plot_reconstruction_error(errors=models_mae_errors,
                                  labels=models_labels,
                                  kin_labels=joint_names,
                                  title=title,
                                  path="media/MUS/pure_reconstruction/"+title+"-%d-cycles.png" % cycles)
        models_labels = []
        models_mae_errors = []


        # figure = plt.figure(figsize=(10, 10))
        # # plot general space distribution
        # z_mean = X_test_transformed
        # print(z_mean.shape)
        # phases = df_test[ExperimentFields.phase.value].values
        # print(phases.shape)
        #
        # z_mean = np.array(z_mean)
        # plt.scatter(z_mean[phases == 2, 0], z_mean[phases == 2, 1], marker='o', c='r', alpha=0.8, label="Manipulation")
        # plt.scatter(z_mean[phases == 3, 0], z_mean[phases == 3, 1], marker='+', c='k', alpha=0.5, label="Retreat")
        # plt.scatter(z_mean[phases == 1, 0], z_mean[phases == 1, 1], marker='2', c='b', alpha=0.5, label="Approach")
        #
        # # Plot user
        # plt.legend(loc='upper left')
        # colors = plt.cm.get_cmap("Dark2", len(traj))
        # i = 0
        # for t in traj:
        #     z_mean = pca.transform(t[[e.value for e in RightHand]].values.astype(np.float32))
        #     subs = t[ExperimentFields.subject.value].values
        #     z_mean = np.array(z_mean)
        #     sub = np.unique(t[ExperimentFields.subject.value])
        #     for s in [1, 6, 10, 8]:
        #         plt.plot(z_mean[subs == s, 0], z_mean[subs == s, 1], '-o', markersize=1.1,
        #                  c=colors(i), lw='0.5', path_effects=[mpe.Stroke(linewidth=1.8, foreground='w'), mpe.Normal()])
        #     i += 1
        # #
        # plt.title("PCA Embedding")
        # plt.savefig(log_dir + "/pca_embedding.png")


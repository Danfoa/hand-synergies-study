
from models.VAE import VariationalAutoEncoder, EmbeddingSpaceLogger, CyclicalAnnealingSchedule, Encoder, Decoder
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorboard.plugins.hparams import api as hp

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error

from utils.data_loader_kine_mus import *



if __name__ == '__main__':
    # Load the dataset
    df = load_subjects_data(DATABASE_PATH, subjects_id=SUBJECTS)
    df.fillna(value=0, inplace=True)

    print(df.shape)


    # Shuffle data for training
    # Split the data into training and testing
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
    del df

    X_train = df_train[[e.value for e in RightHand]].values
    # X_train = StandardScaler(copy=False).fit_transform(X_train)
    X_test = df_test[[e.value for e in RightHand]].values
    # X_test = StandardScaler(copy=False).fit_transform(X_test)

    latent_dim = 2
    original_dim = X_train.shape[1]
    pca = PCA(n_components=latent_dim, svd_solver='full')

    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.transform(X_test)

    X_train_reconstructed = pca.inverse_transform(X_train_transformed)
    X_test_reconstructed = pca.inverse_transform(X_test_transformed)

    joint_names = [e.value for e in RightHand]

    intermediate_dim = 30
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
                        tf.keras.metrics.MeanAbsoluteError(name="MAE_loss"),
                        tf.keras.metrics.MeanSquaredError(name="MSE_loss")])

    batch_size = 2
    epochs = 4
    vae.fit(X_train, X_train,
            epochs=epochs,
            shuffle=False,
            batch_size=batch_size,
            validation_split=0.0,
            workers=4,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()])

    X_train_reconstructed_vae = vae.predict(X_train)
    X_test_reconstructed_vae = vae.predict(X_test)


    error_vector_train = mean_absolute_error(X_train, X_train_reconstructed, multioutput='raw_values')
    print("Train PCA")
    print(error_vector_train)
    print(numpy.mean(error_vector_train))


    error_vector_test = mean_absolute_error(X_test, X_test_reconstructed, multioutput='raw_values')
    print("Test PCA")
    print(error_vector_test)
    print(numpy.mean(error_vector_test))

    error_vector_train = mean_absolute_error(X_train, X_train_reconstructed_vae, multioutput='raw_values')
    print("Train VAE")
    print(error_vector_train)
    print(numpy.mean(error_vector_train))


    error_vector_test = mean_absolute_error(X_test, X_test_reconstructed_vae, multioutput='raw_values')
    print("Test VAE")
    print(error_vector_test)
    print(numpy.mean(error_vector_test))
from utils.data_loader_kine_mus import *
from models.VAE import VariationalAutoEncoder
import tensorflow as tf


from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # Load the dataset
    df = load_subjects_data(DATABASE_PATH, subjects_id=SUBJECTS)

    # Shuffle data for training
    # Split the data into training and testing
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
    del df

    X_train = df_train[[e.value for e in RightHand]].values
    X_test = df_test[[e.value for e in RightHand]].values

    BATCH_SIZE = 1      # Purely stochastic
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(1)
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(1)


    vae = VariationalAutoEncoder(original_dim=18, intermediate_dim=100, latent_dim=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()

    epochs = 10

    # Iterate over epochs.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            # print(step)
            # print(x_batch_train)

            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)  # Add KLD regularization loss

            # Backpropagate gradients through the network
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))

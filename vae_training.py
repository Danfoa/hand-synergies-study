from utils.data_loader_kine_mus import *
from models.VAE import VariationalAutoEncoder, CyclicalAnnealingSchedule, KLLossLogger
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


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
    batch_size = 32
    kl_decay_rate = 0.005
    lr = 1e-3
    intermediate_dim = 50
    latent_dim = 2
    epochs = 5

    train_dataset = tf.data.Dataset.from_tensor_slices(X_train).batch(1)
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(1)

    vae = VariationalAutoEncoder(original_dim=18, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    vae.compile(optimizer,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="reconstruction_MAE")])
    tf.keras.utils.plot_model(vae, 'vae_model.png', show_shapes=True)

    # Callbacks
    log_dir = "models/vae-logs/lr=%.5f-hd=%d-lat_dim=%d-b=%d/" % (lr, intermediate_dim, latent_dim, batch_size)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch',
                                                          profile_batch=0)

    train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')

    vae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, workers=4,
            callbacks=[CyclicalAnnealingSchedule(kl_decay_rate=kl_decay_rate, batch_size=batch_size),
                       tensorboard_callback,
                       KLLossLogger(train_summary_writer, batch_size=batch_size),
                       tf.keras.callbacks.TerminateOnNaN()])
    #
    # mse_loss_fn = tf.keras.losses.MeanSquaredError()
    #
    # loss_metric = tf.keras.metrics.Mean(name='total_loss', dtype=tf.float32)
    # kl_loss_metric = tf.keras.metrics.Mean(name='KL_loss', dtype=tf.float32)
    # reconstruction_loss_metric = tf.keras.metrics.Mean(name='reconstruction_loss', dtype=tf.float32)
    # kl_annealing_metric = tf.keras.metrics.Mean(name='kl_annealing_weight', dtype=tf.float32)
    #
    # # Prepare Tensorboard Summary
    # # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir =
    # # test_log_dir = 'models/vae-logs/' + current_time + '/test'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # # test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    #
    # # Iterate over epochs.
    # for epoch in range(epochs):
    #     print('Start of epoch %d' % (epoch,))
    #
    #     # Iterate over the batches of the dataset.
    #     for batch_num, x_batch_train in enumerate(train_dataset):
    #         # print(step)
    #         # print(x_batch_train)
    #
    #         with tf.GradientTape() as tape:
    #             reconstructed = vae(x_batch_train)
    #             # Compute reconstruction loss
    #             reconstruction_loss = mse_loss_fn(x_batch_train, reconstructed)
    #             # KL loss is calculated inside the `call` method of the VAE
    #             kl_loss = sum(vae.losses)
    #
    #             # Perform KL cost annealing
    #             kl_annealing_weight = 1 / (1 + (1 - 0.00001)/0.00001 * tf.exp(-kl_decay_rate * batch_num * batch_size))
    #
    #             # Add up the two losses
    #             loss = reconstruction_loss + kl_annealing_weight * kl_loss
    #
    #         # Backpropagate gradients through the network
    #         grads = tape.gradient(loss, vae.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    #
    #         # Update metrics
    #         loss_metric(loss)
    #         kl_loss_metric(kl_loss)
    #         reconstruction_loss_metric(reconstruction_loss)
    #         kl_annealing_metric(kl_annealing_weight)
    #
    #         # Log to Tensorboard
    #         with train_summary_writer.as_default():
    #             tf.summary.scalar('total_loss', loss_metric.result(), step=batch_num * batch_size)
    #             tf.summary.scalar('KL_loss', kl_loss_metric.result(), step=batch_num * batch_size)
    #             tf.summary.scalar('reconstruction_loss', reconstruction_loss_metric.result(),
    #                               step=batch_num * batch_size)
    #             tf.summary.scalar('kl_annealing_weight', kl_annealing_metric.result(), step=batch_num * batch_size)
    #
    #         if batch_num % 100 == 0:
    #             print('batch_step %s: mean loss = %s' % (batch_num, loss_metric.result()))
    #
    #     loss_metric.reset_states()
    #     kl_loss_metric.reset_states()
    #     reconstruction_loss_metric.reset_states()

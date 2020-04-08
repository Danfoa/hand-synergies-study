import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import seaborn as sns
import io


class LogAnglePredCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataset, logdir):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.dataset = dataset
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return

        history_angles = None
        ground_truths = None
        for elem in self.dataset.take(1):
            history_angles = elem[0]
            ground_truths = elem[1]

        predictions = self.model.predict(history_angles)
        figure = multi_step_plot(history_angles.numpy()[:, :, 0], ground_truths.numpy(),
                                 predictions,
                                 logdir=self.logdir,
                                 epoch=epoch)
        with self.file_writer.as_default():
            tf.summary.image("Angle Prediction", plot_to_image(figure), step=epoch)
            print("Logging predictions...")


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def multi_step_plot(history, true_future, prediction, logdir, epoch):
    # batch_size = history.shape[0]
    num_rows = 3

    num_in = np.linspace(0, len(history[0, :]) / 0.01, len(history[0, :]))
    num_out = np.linspace((len(history[0, :]) + 1) / 0.01, (len(history[0, :]) + 1 + len(true_future[0, :])) / 0.01,
                          len(true_future[0, :]))

    fig = plt.figure(figsize=(16, 16))
    for row in range(num_rows):

        plt.subplot(num_rows, 3, (row * 3) + 1)
        plt.plot(num_in, np.array(history[(row * 3), :]), label='History')

        plt.plot(num_out, np.array(true_future[(row * 3), :]), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(num_out, np.array(prediction[(row * 3), :]), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.grid("on")

        plt.subplot(num_rows, 3, (row * 3) + 2)
        plt.plot(num_in, np.array(history[(row * 3) + 1, :]), label='History')

        plt.plot(num_out, np.array(true_future[(row * 3) + 1, :]), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(num_out, np.array(prediction[(row * 3) + 1, :]), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.grid("on")

        plt.subplot(num_rows, 3, (row * 3) + 3)
        plt.plot(num_in, np.array(history[(row * 3) + 2, :]), label='History')

        plt.plot(num_out, np.array(true_future[(row * 3) + 2, :]), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(num_out, np.array(prediction[(row * 3) + 2, :]), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.grid("on")

    plt.tight_layout()
    return fig

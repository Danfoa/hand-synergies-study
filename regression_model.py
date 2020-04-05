import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import os
import tensorflow.keras.backend as K


class Regression_Model():

    def r_square(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    # Define metrics to watch
    METRICS = [
        tf.keras.metrics.MeanSquaredError(name='mse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ]




    # Set hyper parameter search
    # HP_HIDDEN_UNITS = hp.HParam('hidden_units', hp.Discrete([10, 50, 200]))
    HP_HIDDEN_UNITS = hp.HParam('hidden_units', hp.Discrete([50]))
    # HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0]))
    HP_HIDDEN_LAYERS = hp.HParam('hidden_layers', hp.Discrete([1, 2]))
    HP_WINDOW_SIZE = hp.HParam('window_size',hp.Discrete([5, 10, 15, 20]))
    HP_RNN = hp.HParam('rnn', hp.Discrete(['vanila', 'gru', 'lstm']))
    # use adam directly
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64]))

    def __init__(self):
        self.configurations = self.__build_configurations()

    def __build_configurations(self):
        configurations = []
        for hl in self.HP_HIDDEN_LAYERS.domain.values:
            for dr in self.HP_DROPOUT.domain.values:
                for rnn in self.HP_RNN.domain.values:
                    for hu in self.HP_HIDDEN_UNITS.domain.values:
                        for lr in self.HP_LEARNING_RATE.domain.values:
                            for bs in self.HP_BATCH_SIZE.domain.values:
                                new = self.__build_conf(hl, dr, rnn, hu, lr, bs)
                                configurations.append(new)
        return configurations

    def __get_rnn_model(self, rnn_model, hidden_layers, dropout, hidden_units):
        lstm_layers = []

        if hidden_layers == 1:
            if rnn_model == 'lstm':
                lstm_layers.append(tf.keras.layers.LSTM(hidden_units, activation='relu', input_shape=(3, 2), name="lstm_0"))
            elif rnn_model == 'gru':
                lstm_layers.append(tf.keras.layers.GRU(hidden_units, activation='relu', input_shape=(3, 2), name="gru_0"))
            else:
                lstm_layers.append(tf.keras.layers.SimpleRNN(hidden_units, activation='relu', input_shape=(3, 2), name="vanila_0"))

        else:
            if rnn_model == 'lstm':
                lstm_layers.append(
                    tf.keras.layers.LSTM(hidden_units, activation='relu', input_shape=(3, 2), return_sequences=True, name="lstm_0"))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_0"))
                lstm_layers.append(
                    tf.keras.layers.LSTM(hidden_units, activation='relu', name="lstm_1"))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_1"))
            elif rnn_model == 'gru':
                lstm_layers.append(
                    tf.keras.layers.GRU(hidden_units, input_shape=(3, 2), activation='relu',return_sequences=True, name="gru_0"))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_0"))
                lstm_layers.append(
                    tf.keras.layers.GRU(hidden_units, activation='relu', name="gru_1"))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_1"))
            else:
                lstm_layers.append(
                    tf.keras.layers.SimpleRNN(hidden_units, input_shape=(3, 2), activation='relu', return_sequences=True,
                                        name="vanila_0"))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_0"))
                lstm_layers.append(
                    tf.keras.layers.SimpleRNN(hidden_units, activation='relu', name="vanila_1"))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_1"))

        model = tf.keras.models.Sequential(name="awsome_net", layers=
            # One or Two LSTM/ GRU layers
            lstm_layers+

            # Output Layer
            [tf.keras.layers.Dropout(dropout),
             tf.keras.layers.Dense(1, name="output_dense_layers")])

        return model

    def __get_callbacks(self, logdir, hparams):
        callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                     tf.keras.callbacks.TensorBoard(logdir,
                                                    update_freq='batch',
                                                    write_graph=False,
                                                    histogram_freq=5),
                     tf.keras.callbacks.EarlyStopping(monitor='val_mse',
                                                      patience=5),
                     hp.KerasCallback(logdir, hparams, trial_id=logdir),
                     tf.keras.callbacks.ModelCheckpoint(
                         filepath=os.path.join(logdir, "checkpoints", "cp.ckpt"),
                         save_best_only=True,
                         monitor='val_mse',
                         verbose=1)
                     ]
        return callbacks


    def __build_conf(self, hl, dr, rnn, hu, lr, bs):
        model = self.__get_rnn_model(rnn_model=rnn, hidden_layers= hl, dropout=dr, hidden_units=hu)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[self.METRICS]
                      )

        # Run log dir
        hparams_log_dir = os.path.join("results", "rnn-hyper-param-search", "logs")
        logdir = os.path.join(hparams_log_dir, "rnn=%s-hl=%d-dr=%d-hu=%d-lr=%s-bs=%d" %
                              (rnn, hl, dr, hu, lr, bs))


        if os.path.exists(logdir):
            pass
            # print("Ignoring run %s" % logdir)
        hparams = {
            self.HP_HIDDEN_LAYERS: hl,
            self.HP_DROPOUT: dr,
            self.HP_RNN: rnn,
            self.HP_HIDDEN_UNITS: hu,
            self.HP_LEARNING_RATE: lr,
            self.HP_BATCH_SIZE: bs
        }
        callbacks = self.__get_callbacks(logdir, hparams)

        return model, callbacks, bs




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
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsolutePercentageError(name='mape')
    ]

    # Set hyper parameter search
    # HP_HIDDEN_UNITS = hp.HParam('hidden_units', hp.Discrete([10, 50, 200]))
    HP_HIDDEN_UNITS = hp.HParam('hidden_units', hp.Discrete([10, 50, 100, 300]))
    # HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.2, 0.4]))
    HP_HIDDEN_LAYERS = hp.HParam('hidden_layers', hp.Discrete([1, 2]))
    HP_WINDOW_SIZE = hp.HParam('window_size', hp.Discrete([3, 5, 10, 15, 20]))
    HP_RNN = hp.HParam('rnn', hp.Discrete(['vanilla', 'gru', 'lstm']))
    # use adam directly
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001, 0.01, 0.0001, 0.00001]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16, 32, 64]))
    HP_SHIFT_PERIOD = hp.HParam('shift_period', hp.Discrete([1, 2, 4, 7]))
    HP_CLIP_GRADIENT = hp.HParam('clip_gradient', hp.Discrete([False, True]))

    def __init__(self, window_size=None, n_features=None, shift_period=None):
        self.shift_period = shift_period
        self.window_size = window_size
        self.n_features = n_features
        self.configurations = self.__build_configurations()

    def __build_configurations(self):
        configurations = []
        for hl in self.HP_HIDDEN_LAYERS.domain.values:
            for dr in self.HP_DROPOUT.domain.values:
                for rnn in self.HP_RNN.domain.values:
                    for hu in self.HP_HIDDEN_UNITS.domain.values:
                        for lr in self.HP_LEARNING_RATE.domain.values:
                            for bs in self.HP_BATCH_SIZE.domain.values:
                                for cg in self.HP_CLIP_GRADIENT.domain.values:
                                    # new = self.__build_conf(hl, dr, rnn, hu, lr, bs)
                                    new = [hl, dr, rnn, hu, lr, bs, cg]
                                    configurations.append(new)
        return configurations

    def __get_rnn_model(self, rnn_model, hidden_layers, dropout, hidden_units):
        lstm_layers = []

        if hidden_layers == 1:
            if rnn_model == 'lstm':
                lstm_layers.append(tf.keras.layers.LSTM(hidden_units,
                                                        activation='relu',
                                                        input_shape=(self.window_size, self.n_features),
                                                        name="lstm_0_%dU" % hidden_units))
            elif rnn_model == 'gru':
                lstm_layers.append(tf.keras.layers.GRU(hidden_units,
                                                       activation='relu',
                                                       input_shape=(self.window_size, self.n_features),
                                                       name="gru_0_%dU" % hidden_units))
            else:
                lstm_layers.append(tf.keras.layers.SimpleRNN(hidden_units,
                                                             activation='relu',
                                                             input_shape = (self.window_size, self.n_features),
                                                             name = "vanilla_0_%dU" % hidden_units))
        else:
            if rnn_model == 'lstm':
                lstm_layers.append(
                    tf.keras.layers.LSTM(hidden_units,
                                         activation='relu',
                                         input_shape=(self.window_size, self.n_features),
                                         return_sequences=True,
                                         name="lstm_0_%dU" % hidden_units))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_0"))
                lstm_layers.append(
                    tf.keras.layers.LSTM(hidden_units, activation='relu', name="lstm_1_%dU" % hidden_units))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_1"))
            elif rnn_model == 'gru':
                lstm_layers.append(
                    tf.keras.layers.GRU(hidden_units,
                                        input_shape=(self.window_size, self.n_features),
                                        activation='relu',
                                        return_sequences=True,
                                        name="gru_0_%dU" % hidden_units))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_0"))
                lstm_layers.append(
                    tf.keras.layers.GRU(hidden_units, activation='relu', name="gru_1_%dU" % hidden_units))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_1"))
            else:
                lstm_layers.append(
                    tf.keras.layers.SimpleRNN(hidden_units,
                                              input_shape=(self.window_size, self.n_features),
                                              activation='relu',
                                              return_sequences=True,
                                        name="vanilla_0_%dU" % hidden_units))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_0"))
                lstm_layers.append(
                    tf.keras.layers.SimpleRNN(hidden_units, activation='relu', name="vanilla_1_%dU" % hidden_units))
                # lstm_layers.append(tf.keras.layers.ReLU(name="input_RELU_1"))

        output_layers = [tf.keras.layers.Dropout(dropout, name="D%.2f" % dropout),
                         tf.keras.layers.Dense(1, name="angle_delta_output")]
                                               # activation=lambda x: 180*tf.keras.activations.tanh(x))]

        model = tf.keras.models.Sequential(name="awsome_net", layers=lstm_layers + output_layers)

        return model

    def __get_callbacks(self, logdir, hparams):
        callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                     tf.keras.callbacks.TensorBoard(logdir,
                                                    update_freq='batch',
                                                    write_graph=False,
                                                    histogram_freq=5),
                     tf.keras.callbacks.EarlyStopping(monitor='val_mse',
                                                      patience=15),
                     hp.KerasCallback(logdir, hparams, trial_id=logdir),
                     tf.keras.callbacks.ModelCheckpoint(
                         filepath=os.path.join(logdir, "cp.ckpt"),
                         save_best_only=True,
                         monitor='val_mse',
                         verbose=1)
                     ]
        return callbacks

    def build_conf(self, hl, dr, rnn, hu, lr, bs, cg):
        model = self.__get_rnn_model(rnn_model=rnn, hidden_layers= hl, dropout=dr, hidden_units=hu)

        if cg:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[self.METRICS]
                          )
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[self.METRICS]
                          )
        # Run log dir
        hparams_log_dir = os.path.join("/content/drive/", "My Drive", "rnn-hyper-param-search", "logs")
        hparams_writer = tf.summary.create_file_writer(hparams_log_dir)
        with hparams_writer.as_default():
            hp.hparams_config(hparams=[self.HP_HIDDEN_LAYERS,
                                      self.HP_DROPOUT,
                                      self.HP_RNN,
                                      self.HP_HIDDEN_UNITS,
                                      self.HP_LEARNING_RATE,
                                      self.HP_BATCH_SIZE,
                                      self.HP_WINDOW_SIZE,
                                      self.HP_SHIFT_PERIOD,
                                      self.HP_CLIP_GRADIENT],
                metrics=[
                    hp.Metric('epoch_loss', group="train", display_name='epoch_loss'),
                    hp.Metric('epoch_loss', group="validation", display_name='val_loss'),
                    hp.Metric('mape', group="train", display_name='mape'),
                    hp.Metric('ecpoh_mape', group="validation", display_name='val_mape'),
                    hp.Metric('mae', group="train", display_name='mae'),
                    hp.Metric('epoch_mae', group="validation", display_name='val_mae'),
                    hp.Metric('rmse', group="train", display_name='rmse'),
                    hp.Metric('epoch_rmse', group="validation", display_name='val_rmse')
                ])

        # hparams_log_dir = os.path.join("results", "rnn-hyper-param-search", "logs")
        logdir = os.path.join(hparams_log_dir, "rnn=%s-hl=%d-dr=%d-hu=%d-lr=%s-bs=%d-ws-%d-sp=%d-cg=%s" %
                              (rnn, hl, dr, hu, lr, bs, self.window_size, self.shift_period, cg))

        if os.path.exists(logdir):
            pass
            # print("Ignoring run %s" % logdir)
        hparams = {
            self.HP_HIDDEN_LAYERS: hl,
            self.HP_DROPOUT: dr,
            self.HP_RNN: rnn,
            self.HP_HIDDEN_UNITS: hu,
            self.HP_LEARNING_RATE: lr,
            self.HP_BATCH_SIZE: bs,
            self.HP_WINDOW_SIZE: self.window_size,
            self.HP_SHIFT_PERIOD: self.shift_period,
            self.HP_CLIP_GRADIENT: cg
        }
        callbacks = self.__get_callbacks(logdir, hparams)

        return model, callbacks, bs

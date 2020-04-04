from models.VAE import VariationalAutoEncoder, EmbeddingSpaceLogger, CyclicalAnnealingSchedule, Encoder, Decoder
import tensorflow as tf
import os

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

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == '__main__':
    # Load the dataset
    df = load_subjects_data(DATABASE_PATH, subjects_id=SUBJECTS)
    df.fillna(value=0, inplace=True)

    # _________________________________________________________________
    # Split the data into training and testing
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    # del df

    train_scaler = StandardScaler()
    X_train = df_train[[e.value for e in RightHand]].values
    X_train_scaled = train_scaler.fit_transform(X_train)

    test_scaler = StandardScaler()
    X_test = df_test[[e.value for e in RightHand]].values
    X_test_scaled = test_scaler.fit_transform(X_test)

    # _________________________________________________________________
    joint_names = [e.value for e in RightHand]
    joint_delta_names = [e.value + "_dt" for e in RightHand]
    sEMG_names = [e.value for e in sEMG]
    num_emgs = len(sEMG_names)

    # _________________________________________________________________

    # for adl_id in ADLs:
    #     test_adl = df[df[ExperimentFields.adl_ids.value] == adl_id]
    #     test_adl = test_adl[test_adl[ExperimentFields.subject.value] > 15]
    #     shifted = test_adl.shift(periods=-1, axis=0)
    #
    #     result = df - shifted

    # Create data for sEMG control with `td` time lag
    # td = 0.01
    # df = test_adl
    # shift_periods = 1
    # joint_values = df[joint_names].values
    #
    # delta_movements = joint_values[shift_periods:, :] - joint_values[:-shift_periods, :]
    # shifted_df = df.shift(periods=-shift_periods)
    # shifted_df.reset_index(inplace=True, drop=True)
    # df_copy = df.copy()
    # df_copy[joint_delta_names] = pd.DataFrame(df[joint_names].values - shifted_df[joint_names].values,
    #                                           index=df.index)
    # sEMG_activations = df[sEMG_names].values[:-shift_periods]
    # target_positions = joint_values
    #
    # # Build the entire prediction model _____________________________________________
    # emg_inputs = tf.keras.Input(shape=(num_emgs,), name='sEMG_activations')
    # kin_inputs = tf.keras.Input(shape=(len(joint_names),), name='sEMG_activations')
    #
    # z_mean, z_log_var, _ = vae.encoder(kin_inputs)
    #
    # # Prediction part
    # lstm_pred = tf.keras.layers.LSTM(latent_dim*10)(emg_inputs)


    # prediction_model = tf.keras.Model(inputs=[emg_inputs, kin_inputs], outputs=[pred_joint_state])
    #_____________________________________

    tf.keras.utils.plot_model(prediction_model, to_file="prediction_model", show_shapes=True)






    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)

    joint_names = [e.value for e in RightHand]

    # test_adl = df[df[ExperimentFields.adl_ids.value] == 26]
    # test_adl = test_adl[test_adl[ExperimentFields.subject.value] == 1]

    emg_labels = [e.value for e in sEMG]
    test_adl = pd.melt(test_adl,
                       id_vars=[e.value for e in ExperimentFields],
                       value_vars=emg_labels,
                       var_name="sEMG",
                       value_name="emg_activation")
    sn_plot = sns.relplot(x=ExperimentFields.time.value,
                          y="emg_activation",
                          hue=ExperimentFields.phase.value,
                          col="sEMG",
                          units=ExperimentFields.subject.value, estimator=None,
                          col_wrap=4,
                          legend=False,
                          data=test_adl,
                          kind='line',
                          markers=True)
    sn_plot.fig.suptitle("ADL %d - ALL Subjects" % adl_id, x=0.87, y=0.25, fontsize=20)
    # plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="lower right", borderaxespad=0.)
    plt.savefig("media/MUS/sEMG/ADL%d.png" % adl_id)
    plt.show()
    # break

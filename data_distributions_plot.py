from utils.data_loader_kine_mus import *

from models.VAE import VariationalAutoEncoder, EmbeddingSpaceLogger, CyclicalAnnealingSchedule, Encoder, Decoder
import tensorflow as tf

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # Load the dataset
    df = load_subjects_data(DATABASE_PATH, subjects_id=SUBJECTS)

    # Shuffle data for training
    # Split the data into training and testing
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
    del df

    df_plot = pd.melt(df_train,
                      id_vars=[e.value for e in ExperimentFields],
                      value_vars=[e.value for e in RightHand],
                      var_name="joint_id",
                      value_name="angle")

    s_plot = sns.catplot(x="joint_id", y="angle", kind="violin", scale="count", linewidth=0.9, cut=0,
                         aspect=2,
                         data=df_plot,
                         orient='v')
    s_plot.set_xticklabels(rotation=45)
    plt.title("KINE-MUS Joint Values")
    plt.tight_layout()
    plt.show()

    # Plot EMG data
    df_plot = pd.melt(df_train,
                      id_vars=[e.value for e in ExperimentFields],
                      value_vars=[e.value for e in sEMG],
                      var_name="sEMG",
                      value_name="Activation")

    s_plot = sns.catplot(x="sEMG", y="Activation", kind="violin", scale="count", linewidth=0.9, cut=0,
                         aspect=1.5,
                         data=df_plot,
                         orient='v')
    s_plot.set_xticklabels(rotation=45)
    plt.title("KINE-MUS EMG Activations")
    plt.tight_layout()
    plt.show()

import tensorflow as tf
import numpy as np
import pandas as pd
from utils.data_loader_kine_mus import ExperimentFields, DATABASE_PATH, SUBJECTS, ADLs, RightHand, sEMG, load_subject_data, \
    load_subjects_data
import pprint
import matplotlib.pyplot as plt


def get_windowed_dataset(dataset_path, target_size, window_size, target_joints, shift_periods=1, subjects=SUBJECTS,
                         adl_ids=ADLs, phase_ids=None):
    joint_names = [e.value for e in RightHand]
    joint_delta_names = [e.value + "_dt" for e in RightHand]
    sEMG_names = [e.value for e in sEMG]
    experiment_fields = [e.value for e in ExperimentFields]
    num_emgs = len(sEMG_names)

    min_dt, max_dt = 10, -10
    dataset = None
    print("Loading data from subject: ", end="")
    for subject in subjects:
        print(subject, end=",")
        for adl in adl_ids:
            if adl == 25 and subject == 21:
                continue
            # print("Loading data drom subject %d, adl %d and phase %d" % (subject, adl, phase))
            df = load_subject_data(dataset_path, subject_id=subject, adl_ids=[adl], phase_ids=phase_ids)

            df = df[target_joints + sEMG_names]
            # Configure TF dataset
            tmp_dataset = tf.data.Dataset.from_tensor_slices(df.values)
            tmp_dataset = tmp_dataset.window(window_size + target_size, shift=1, stride=1, drop_remainder=True)
            tmp_dataset = tmp_dataset.flat_map(lambda x: x.batch(window_size + target_size))
            tmp_dataset = tmp_dataset.map(lambda x: (x[:-target_size, :],
                                                     x[-target_size:, 0]),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if dataset is None:
                dataset = tmp_dataset
            else:
                dataset = dataset.concatenate(tmp_dataset)
    return dataset


def create_time_steps(length):
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = np.linspace(0, len(history) / 0.01, len(history))
    num_out = np.linspace((len(history) + 1) / 0.01, (len(history) + 1 + len(true_future)) / 0.01, len(true_future))

    plt.plot(num_in, np.array(history), label='History')

    plt.plot(num_out, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(num_out, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":

    # Which joints want to be keept from the data
    target_joints = ["WA"]

    dataset = get_windowed_dataset(dataset_path=DATABASE_PATH, target_size=10, window_size=30,
                                   target_joints=target_joints,
                                   shift_periods=1, subjects=SUBJECTS, adl_ids=ADLs, phase_ids=None)

    df = load_subjects_data(database_path=DATABASE_PATH, subjects_id=SUBJECTS, adl_ids=ADLs,
                            phase_ids=None)
    count = 0
    for _ in dataset:
        count += 1

    print(count)
    print(df.shape[0])
    # data
    for x, y in dataset.take(1):
        print(x.shape)
        print(y.shape)
        x = x.numpy()
        y = y.numpy()
        multi_step_plot(x[:, 0], y, np.array([0]))

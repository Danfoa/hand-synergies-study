import tensorflow as tf
import pandas as pd
from .data_loader_kine_mus import ExperimentFields, DATABASE_PATH, SUBJECTS, ADLs, RightHand, sEMG, load_subject_data
import pprint

def get_windowed_dataset(dataset_path, target_size, window_size, target_joints, shift_periods=1, subjects=SUBJECTS, adl_ids=ADLs, phase_ids=None):
def get_windowed_dataset(dataset_path, window_size, target_joints, shift_periods=1, subjects=SUBJECTS, adl_ids=ADLs, phase_ids=None):
    joint_names = [e.value for e in RightHand]
    joint_delta_names = [e.value + "_dt" for e in RightHand]
    sEMG_names = [e.value for e in sEMG]
    experiment_fields = [e.value for e in ExperimentFields]
    num_emgs = len(sEMG_names)

    dataset = None
    for subject in subjects:
        for adl in adl_ids:
            for phase in phase_ids:
                # print("Loading data drom subject %d, adl %d and phase %d" % (subject, adl, phase))
            df = load_subject_data(dataset_path, subject_id=subject, adl_ids=[adl], phase_ids=phase_ids)
                # Find the angle deltas
                shifted_df = df.shift(periods=-shift_periods)
                shifted_df.reset_index(inplace=True, drop=True)
                df[joint_delta_names] = pd.DataFrame(shifted_df[joint_names].values - df[joint_names].values, index=df.index)
                df.drop(df.tail(shift_periods).index, inplace=True)

            df = df[target_joints + sEMG_names]
                target_deltas = [d + "_dt" for d in target_joints]
                df = df[target_joints + target_deltas + sEMG_names]

                # Configure TF dataset
                tmp_dataset = tf.data.Dataset.from_tensor_slices(df.values)
            tmp_dataset = tmp_dataset.window(window_size + target_size, shift=1, stride=1, drop_remainder=True)
            tmp_dataset = tmp_dataset.flat_map(lambda x: x.batch(window_size + target_size))
            tmp_dataset = tmp_dataset.map(lambda x: (x[:-target_size, :],
                                                     x[-target_size:, 0]),
                                                 "sEMG": x[:, 2:]},
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
                if dataset is None:
                    dataset = tmp_dataset
                else:
                    dataset = dataset.concatenate(tmp_dataset)
    return dataset


def create_time_steps(length):
  return list(range(-length, 0))

    return dataset


if __name__ == "__main__":

    # Which joints want to be keept from the data
    target_joints = ["WA"]

    dataset = get_windowed_dataset(dataset_path=DATABASE_PATH, target_size=10, window_size=30, target_joints=target_joints,
                                   shift_periods=1, subjects=[1], adl_ids=ADLs, phase_ids=None)

    df = load_subject_data(database_path=DATABASE_PATH, subject_id=1, adl_ids=ADLs,
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
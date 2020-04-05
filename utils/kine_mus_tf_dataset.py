import tensorflow as tf
import pandas as pd
from .data_loader_kine_mus import ExperimentFields, DATABASE_PATH, SUBJECTS, ADLs, RightHand, sEMG, load_subject_data
import pprint


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
                df = load_subject_data(dataset_path, subject_id=subject, adl_ids=[adl], phase_ids=[phase])
                # Find the angle deltas
                shifted_df = df.shift(periods=-shift_periods)
                shifted_df.reset_index(inplace=True, drop=True)
                df[joint_delta_names] = pd.DataFrame(shifted_df[joint_names].values - df[joint_names].values, index=df.index)
                df.drop(df.tail(shift_periods).index, inplace=True)

                #
                target_deltas = [d + "_dt" for d in target_joints]
                df = df[target_joints + target_deltas + sEMG_names]

                # Configure TF dataset
                tmp_dataset = tf.data.Dataset.from_tensor_slices(df.values)
                tmp_dataset = tmp_dataset.window(window_size, drop_remainder=True)
                tmp_dataset = tmp_dataset.flat_map(lambda x: x.batch(window_size))
                tmp_dataset = tmp_dataset.map(lambda x: {"angles_delta": x[-1, len(target_joints):len(target_joints)+len(target_deltas)],
                                                 "angles": x[-1, :len(target_joints)],
                                                 "sEMG": x[:, 2:]},
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
                if dataset is None:
                    dataset = tmp_dataset
                else:
                    dataset = dataset.concatenate(tmp_dataset)

    return dataset


if __name__ == "__main__":

    # Which joints want to be keept from the data
    target_joints = ["WA"]

    dataset = get_windowed_dataset(dataset_path=DATABASE_PATH, window_size=10, target_joints=target_joints,
                                   shift_periods=1, subjects=[1], adl_ids=[1], phase_ids=[1, 3])

    for window in dataset:
        pprint.pprint(window)
        break

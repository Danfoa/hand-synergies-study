import numpy
import scipy
import pandas

from sklearn.decomposition import PCA

from utils.visualization import *
from utils.data_loader_kine_adl import *
import os


DATABASE_PATH = 'kine-adl-be-uji_dataset/CSV DATA'


def generate_recordings_pca(subjects, records, experiment, tasks=None, n_components=10):
    df = load_static_grasps(database_path=DATABASE_PATH,
                            subjects_id=subjects,
                            experiment_number=experiment,
                            records_id=records)

    print("Analysing records with shape ", df.shape)

    # Obtain stable grasps for each hand.
    right_hand_stable_grasps = df[[e.value for e in RightHand]]
    left_hand_stable_grasps = df[[e.value for e in LeftHand]]

    # Remove possible NANs values.
    right_hand_stable_grasps = remove_timesteps_with_missing_values(right_hand_stable_grasps)
    left_hand_stable_grasps = remove_timesteps_with_missing_values(left_hand_stable_grasps)

    # Apply PCA to left hand
    left_hand_pca = PCA(n_components=n_components)
    left_hand_pca.fit_transform(left_hand_stable_grasps)
    # Apply PCA to right hand
    right_hand_pca = PCA(n_components=n_components)
    right_hand_pca.fit_transform(right_hand_stable_grasps)

    # Save data into figure
    title = 'PCA on %d stable grasps - KINE_ADL_BE_UJI dataset \nsubjects:(%d-%d)-records:(%d-%d)' \
            % (left_hand_stable_grasps.shape[0],
               min(subjects), max(subjects),
               min(records), max(records))
    path = 'media/pca/E%d/pca-sub_(%d-%d)-Tasks_(%d).png' % (experiment,
                                                                  min(subjects), max(subjects),
                                                                  # min(records), max(records),
                                                                             tasks[0])

    plot_pca_variances(left_hand_pca, right_hand_pca, title=title, save_path=path)


if __name__ == '__main__':

    # Analyse the entire experiment 1 stable grasps
    subjects = EXP1_SUBJECTS
    records = EXP1_RECORDS
    experiment = 1
    for task in EXP1_TASKS:
        generate_recordings_pca(subjects=subjects, records=records, tasks=[task], experiment=experiment, n_components=10)

    # Analyse the entire experiment 2 stable grasps
    subjects = EXP2_SUBJECTS
    records = EXP2_RECORDS
    experiment = 2
    for task in EXP2_TASKS:
        generate_recordings_pca(subjects=subjects, records=records, tasks=[task], experiment=experiment, n_components=10)
    generate_recordings_pca(subjects=subjects, records=records, experiment=experiment, n_components=10)

    # # Deformable objects ids: [31, 32 33, 26, 43]
    # # Analyse the entire experiment 1 stable grasps on deformable objects
    # subjects = EXP2_SUBJECTS
    # records = [201, 207, 209, 210]
    # tasks = [1, 2, 19, 20, 21, 28, 30, 33, 34]
    # experiment = 2
    # generate_recordings_pca(subjects=subjects, records=records, tasks=tasks, experiment=experiment, n_components=10)
    #

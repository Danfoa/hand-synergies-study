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
                            records_id=records,
                            tasks_id=tasks)
                    

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
    subjects_string = "sub=" + str(subjects) if len(subjects) == 1 else "sub=(%d-%d)" % (min(subjects), max(subjects))
    tasks_string = "task=" + str(tasks) if len(tasks) == 1 else "tasks=(%d-%d)" % (min(tasks), max(tasks))
    records_string = "record=" + str(records) if len(records) == 1 else "records=(%d-%d)" % (min(records), max(records))

    title = ('PCA on %d stable grasps - KINE_ADL_BE_UJI dataset \n' % left_hand_stable_grasps.shape[0])  + subjects_string + ' - ' + records_string + ' - ' + tasks_string
    path = ('media/pca/E%d/pca-' %  experiment) + subjects_string + ' - ' + records_string + ' - ' + tasks_string
                                                         
    plot_pca_variances(left_hand_pca, right_hand_pca, title=title, save_path=path + '.png')

    # Save eigenvalues and eigenvectors of experiment 
    numpy.savez(path, right_eigenvectors=right_hand_pca.components_, right_eigenvalues=right_hand_pca.explained_variance_, 
                left_eigenvectors=left_hand_pca.components_, left_eigenvalues=left_hand_pca.explained_variance_)

if __name__ == '__main__':

    # Compute task specific PC's across all subjects. -------------------------------------------------------------------
    # # Analyse the entire experiment 1 stable grasps
    # subjects = EXP1_SUBJECTS
    # records = EXP1_RECORDS
    # experiment = 1
    # for task in EXP1_TASKS:
    #     generate_recordings_pca(subjects=subjects, records=records, tasks=[task], experiment=experiment, n_components=10)

    # # Analyse the entire experiment 2 stable grasps
    # subjects = EXP2_SUBJECTS
    # records = EXP2_RECORDS
    # experiment = 2
    # for task in EXP2_TASKS:
        # generate_recordings_pca(subjects=subjects, records=records, tasks=[task], experiment=experiment, n_components=10)

    # Compute Subject specific PC's -------------------------------------------------------------------------------------
    # Analyse the entire experiment 1 stable grasps
    # subjects = EXP1_SUBJECTS
    # records = EXP1_RECORDS
    # experiment = 1
    # tasks = EXP1_TASKS
    # for subject in EXP1_SUBJECTS:
    #     generate_recordings_pca(subjects=[subject], records=records, tasks=tasks, experiment=experiment, n_components=10)

    # Analyse the entire experiment 2 stable grasps
    records = EXP2_RECORDS
    experiment = 2
    tasks = EXP2_TASKS
    for subject in EXP2_SUBJECTS:
        generate_recordings_pca(subjects=[subject], records=records, tasks=tasks, experiment=experiment, n_components=10)

   
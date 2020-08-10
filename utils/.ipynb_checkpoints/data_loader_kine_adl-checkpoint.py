
import os
import pandas
import numpy
import matplotlib.pyplot as plt
from enum import Enum

# from utils.visualization import *

DATABASE_PATH = 'kine-adl-be-uji_dataset/CSV DATA'

EXP1_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
EXP1_RECORDS = list(range(101, 134))
EXP1_TASKS = list(range(1, 100))

EXP2_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25]
EXP2_RECORDS = list(range(201, 226))
EXP2_TASKS = list(range(1, 80))


class ExperimentFields(Enum):
    experiment = 'EXPERIMENT'
    subject = 'SUBJECT'
    record_id = 'R'
    task_id = 'ID '
    time = 'TIME'


class RightHand(Enum):
    cmc1_f = 'R_CMC1_F'
    cmc1_a = 'R_CMC1_A'
    mpc1_f = 'R_MCP1_F'
    ip1_f = 'R_IP1_F'
    mcp2_f = 'R_MCP2_F'
    mcp23_a = 'R_MCP23_A'
    pip2_f = 'R_PIP2_F'
    mcp3_f = 'R_MCP3_F'
    pip3_f = 'R_PIP3_F'
    mcp4_f = 'R_MCP4_F'
    mcp34_a = 'R_MCP34_A'
    pip4_f = 'R_PIP4_F'
    palm_arch = 'R_PalmArch'
    mcp5_f = 'R_MCP5_F'
    mcp45_a = 'R_MCP45_A'
    pip5_f = 'R_PIP5_F'
    wr_f = 'R_WR_F'
    wr_a = 'R_WR_A'


class LeftHand(Enum):
    cmc1_f = 'L_CMC1_F'
    cmc1_a = 'L_CMC1_A'
    mpc1_f = 'L_MCP1_F'
    ip1_f = 'L_IP1_F'
    mcp2_f = 'L_MCP2_F'
    mcp23_a = 'L_MCP23_A'
    pip2_f = 'L_PIP2_F'
    mcp3_f = 'L_MCP3_F'
    pip3_f = 'L_PIP3_F'
    mcp4_f = 'L_MCP4_F'
    mcp34_a = 'L_MCP34_A'
    pip4_f = 'L_PIP4_F'
    palm_arch = 'L_PalmArch'
    mcp5_f = 'L_MCP5_F'
    mcp45_a = 'L_MCP45_A'
    pip5_f = 'L_PIP5_F'
    wr_f = 'L_WR_F'
    wr_a = 'L_WR_A'


def load_subjects_data(database_path, subjects_id, experiment_number=1,
                       task_id=None, record_id=None):
    df = pandas.DataFrame()

    for subject_id in subjects_id:
        df = pandas.concat((df, load_subject_data(database_path=database_path,
                                                  subject_id=subject_id,
                                                  experiment_number=experiment_number,
                                                  task_id=task_id,
                                                  records_id=record_id)))
        df.reset_index(drop=True, inplace=True)
    return df


def load_subject_data(database_path, subject_id, experiment_number=1,
                      task_id=None, records_id=None, load_anatomic_data=True):
    file_name = '/E%d/KINEMATIC_DATA_E%d_S%d.csv' % (experiment_number,
                                                     experiment_number,
                                                     subject_id)
    
    
    
        
    df = pandas.read_csv(filepath_or_buffer=database_path + file_name)

    # Filter data by record number/s
    if isinstance(records_id, list) and records_id is not None:
        combined_df = pandas.DataFrame(columns=df.columns)
        for rec_num in records_id:
            combined_df = pandas.concat((combined_df, df[df[ExperimentFields.record_id.value] == rec_num]),
                                        axis=0)
        combined_df.reset_index(drop=True, inplace=True)
        df = combined_df

    # Filter data by task id/s
    if isinstance(task_id, list) and task_id is not None:
        combined_df = pandas.DataFrame(columns=df.columns)
        for id in task_id:
            combined_df = pandas.concat((combined_df, df[df[ExperimentFields.task_id.value] == id]), axis=0)
        combined_df.reset_index(inplace=True)
        df = combined_df
    
    if load_anatomic_data:
        anatomic_labels = ['HL_R','HL_L','HW_R','HW_L']

        df_anatomic = pandas.read_csv(os.path.join(database_path, "SUBJECT_DATA.csv"))
        # Load single subject anatomic data
        df_anatomic = df_anatomic[df_anatomic[ExperimentFields.subject.value] == subject_id] 
        # Load only hand size measurements
        df_anatomic = df_anatomic[anatomic_labels]
        data = numpy.multiply(numpy.ones((df.shape[0], len(anatomic_labels))), df_anatomic.values)
        df_anatomic = pandas.DataFrame(data=data, columns=anatomic_labels)

        df = pandas.concat([df, df_anatomic], axis=1, copy=False)


    return df


def load_static_grasps(database_path, subjects_id, records_id, experiment_number=1, tasks_id=None):
    print("Loading subjects stable initial grasps:\nSubject: ")
    df = None
    for subject_id in subjects_id:
        print(subject_id)
        for record_id in records_id:
            subject_data = load_subject_data(database_path=database_path,
                                             subject_id=subject_id,
                                             experiment_number=experiment_number,
                                             records_id=[record_id],
                                             task_id=tasks_id)
            if df is None:
                df = pandas.DataFrame(columns=subject_data.columns)

            # Get id of the task performed in each record
            tasks_ids = numpy.unique(subject_data[ExperimentFields.task_id.value].values)
            for task_id in tasks_ids:
                task_data = subject_data[subject_data[ExperimentFields.task_id.value] == task_id]
                # Save first joint possitions since they represent the stable grasp
                # print(task_data.shape)
                df = pandas.concat((df, task_data.iloc[[0], :]), axis=0)
                # print(df.shape)
    print('')
    df.reset_index(drop=True, inplace=True)
    return df


def remove_timesteps_with_missing_values(df):
    data_with_missing_values = df.index[df.isna().any(axis=1)]
    if len(data_with_missing_values) > 0:
        print("Removing %.2f%% observations with missing numerical values" %
              ((len(data_with_missing_values) / df.shape[0]) * 100))
        df = df.drop(index=data_with_missing_values)
    return df
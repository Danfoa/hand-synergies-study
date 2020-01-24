import pandas
import numpy
import matplotlib.pyplot as plt
from enum import Enum

from utils.visualization_MUS import *

DATABASE_PATH = 'kine-mus-uji-dataset/CSV_DATA/'

EXP1_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
EXP1_RECORDS = list(range(101, 134))
EXP1_TASKS = list(range(1, 100))

EXP2_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25]
EXP2_RECORDS = list(range(201, 226))
EXP2_TASKS = list(range(1, 80))

class ExperimentFields(Enum):
    phase = 'Phase'
    adl_id = 'ADL'
    time = 'time'

class sEMG(Enum):
    sEMG_1 = "sEMG_1"
    sEMG_2 = "sEMG_2" 
    sEMG_3 = "sEMG_3" 
    sEMG_4 = "sEMG_4" 
    sEMG_5 = "sEMG_5" 
    sEMG_6 = "sEMG_6" 
    sEMG_7 = "sEMG_7" 



class RightHand(Enum):
    cmc1_f = 'CMC1_F'
    cmc1_a = 'CMC1_A'
    mpc1_f = 'MCP1_F'
    ip1_f = 'IP1_F'
    mcp2_f = 'MCP2_F'
    mcp23_a = 'MCP23_A'
    pip2_f = 'PIP2_F'
    mcp3_f = 'MCP3_F'
    pip3_f = 'PIP3_F'
    mcp4_f = 'MCP4_F'
    mcp34_a = 'MCP34_A'
    pip4_f = 'PIP4_F'
    palm_arch = 'PalmArch'
    mcp5_f = 'MCP5_F'
    mcp45_a = 'MCP45_A'
    pip5_f = 'PIP5_F'
    wr_f = 'WR_F'
    wr_a = 'WR_A'

def load_subjects_data(database_path, subjects_id, task_id=None, record_id=None):
    df = pandas.DataFrame()

    for subject_id in subjects_id:
        df = pandas.concat((df, load_subject_data(database_path=database_path,
                                                  subject_id=subject_id,
                                                  experiment_number=experiment_number,
                                                  task_id=task_id,
                                                  records_id=record_id)))
        df.reset_index(drop=True, inplace=True)
    return df


def load_subject_data(database_path, subject_id, adl_id=None):
    file_name = 'KIN_MUS_S%d.csv' % (subject_id)
    df = pandas.read_csv(filepath_or_buffer = database_path + file_name)

    # Filter data by task id/s
    if isinstance(adl_id, list):
        combined_df = pandas.DataFrame(columns=df.columns)
        for id in adl_id:
            combined_df = pandas.concat((combined_df, df[df[ExperimentFields.adl_id.value] == id]), axis=0)
        combined_df.reset_index(inplace=True)
        df = combined_df

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
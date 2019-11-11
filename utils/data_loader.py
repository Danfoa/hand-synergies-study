import pandas
import numpy
import matplotlib.pyplot as plt
from enum import Enum

from utils.visualization import *

EXP1_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
EXP1_RECORDS = list(range(101, 134))

EXP2_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25]
EXP2_RECORDS = list(range(201, 226))

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
    # wr_f = 'R_WR_F'   # Data missing in exp 1
    # wr_a = 'R_WR_A'   # Data missing in exp 1


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
    # wr_f = 'L_WR_F' # Data missing in exp 1
    # wr_a = 'L_WR_A' # Data missing in exp 1


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
                      task_id=None, records_id=None):
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

    return df


def load_static_grasps(database_path, subjects_id, records_id, experiment_number=1):
    print("Loading subjects stable initial grasps:\nSubject: ")
    df = None
    for subject_id in subjects_id:
        print(subjects_id, end=', ')
        for record_id in records_id:
            subject_data = load_subject_data(database_path=database_path,
                                             subject_id=subject_id,
                                             experiment_number=experiment_number,
                                             records_id=[record_id])
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



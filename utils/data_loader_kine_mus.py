import pandas
import numpy
import matplotlib.pyplot as plt
from enum import Enum


DATABASE_PATH = 'kine-mus-uji-dataset/CSV_Data/'

SUBJECTS = list(range(1, 23))
ADLs = list(range(1, 27))


class ExperimentFields(Enum):
    phase = 'Phase'
    adl_ids = 'ADL'
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
    wr_f = 'WF'
    wr_a = 'WA'


def load_subjects_data(database_path, subjects_id, adl_ids=None, phase_ids=None):
    df = pandas.DataFrame()
    for subject_id in subjects_id:
        df = pandas.concat((df, load_subject_data(database_path=database_path,
                                                  subject_id=subject_id,
                                                  adl_ids=adl_ids,
                                                  phase_ids=phase_ids)))
        df.reset_index(drop=True, inplace=True)
    return df


def load_subject_data(database_path, subject_id, adl_ids=None, phase_ids=None):
    file_name = 'KIN_MUS_S%d.csv' % subject_id
    df = pandas.read_csv(filepath_or_buffer=database_path + file_name, dtype=numpy.float32)
        
    # Filter data by task id/s
    if isinstance(adl_ids, list):
        combined_df = pandas.DataFrame(columns=df.columns)
        for id in adl_ids:
            combined_df = pandas.concat((combined_df, df[df[ExperimentFields.adl_ids.value] == id]), axis=0)
        combined_df.reset_index(inplace=True)
        df = combined_df

    # Filter data by phase id/s
    if isinstance(phase_ids, list):
        combined_df = pandas.DataFrame(columns=df.columns)
        for phase in phase_ids:
            combined_df = pandas.concat((combined_df, df[df[ExperimentFields.phase.value] == phase]), axis=0)
        combined_df.reset_index(inplace=True)
        df = combined_df
        
    return df


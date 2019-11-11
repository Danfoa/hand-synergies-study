import pandas
import numpy
import matplotlib.pyplot as plt

from utils.data_loader import ExperimentFields
from utils.data_loader import RightHand
from utils.data_loader import LeftHand


def plot_task_data(df):

    subjects = numpy.unique(df[ExperimentFields.subject.value].values)
    experiment = numpy.unique(df[ExperimentFields.experiment.value].values)
    tasks = numpy.unique(df[ExperimentFields.task_id.value].values)
    records = numpy.unique(df[ExperimentFields.record_id.value].values)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
    fig.set_size_inches(15, 12)
    # fig.set_size_
    ax1.plot(df[ExperimentFields.time.value], df[[e.value for e in RightHand]], '-')
    ax1.set_title('Right Hand')
    ax1.set_ylabel('Angle position [deg]')
    ax1.grid()
    ax1.legend([e.value for e in RightHand])
    ax2.plot(df[ExperimentFields.time.value], df[[e.value for e in LeftHand]], '-')
    ax2.set_title('Left Hand')
    ax2.set_ylabel('Angle position [deg]')
    ax2.set_xlabel('Time [s]')
    ax2.grid()
    ax2.legend([e.value for e in LeftHand])

    fig.suptitle("Subject: %s - Experiment: %s \n Record: %s - Task: %s" %
                 (subjects, experiment, records, tasks))
    plt.autoscale(True)
    plt.tight_layout(pad=5, w_pad=1, h_pad=3.0)
    plt.show(block=False)

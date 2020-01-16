import pandas
import numpy
import matplotlib.pyplot as plt

from utils.data_loader_kine_adl import ExperimentFields
from utils.data_loader_kine_adl import RightHand
from utils.data_loader_kine_adl import LeftHand


def plot_pca_variances(left_pca, right_pca, title, save_path=None):
    plt.ioff()

    n_components_left = left_pca.n_components_
    n_components_right = right_pca.n_components_

    # PRINT EIGEN VALUES VARIANCES
    labels_left = ['PC%d' % (c + 1) for c in range(n_components_left)]
    labels_right = ['PC%d' % (c + 1) for c in range(n_components_right)]

    x_left = numpy.arange(n_components_left)  # the label locations
    x_right = numpy.arange(n_components_right)  # the label locations

    width = 0.2  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='all')
    # for t in thresholds:
    ax1.bar(x_left, left_pca.explained_variance_ratio_ * 100, width)
    ax2.bar(x_right, right_pca.explained_variance_ratio_ * 100, width)
    plt.pause(0.05)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Variance explained [%]')
    ax2.set_ylabel('Variance explained [%]')
    ax1.set_xlabel('Components')
    ax2.set_xlabel('Components')
    ax1.set_title("Left hand")
    ax2.set_title("Right hand")

    ax1.set_xticks(x_left)
    ax2.set_xticks(x_right)
    ax1.set_yticks(numpy.arange(0, 80, 10))
    ax2.set_yticks(numpy.arange(0, 80, 10))
    ax1.set_xticklabels(labels_left)
    ax2.set_xticklabels(labels_right)
    ax1.grid()
    ax2.grid()

    plt.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    # else:
    #     plt.draw()
    #     plt.pause(0.001)
    # plt.show()


def plot_task_data(df, block=False):

    if not block:
        plt.ion()
    subjects = numpy.unique(df[ExperimentFields.subject.value].values)
    experiment = numpy.unique(df[ExperimentFields.experiment.value].values)
    tasks = numpy.unique(df[ExperimentFields.task_id.value].values)
    records = numpy.unique(df[ExperimentFields.record_id.value].values)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
    fig.set_size_inches(17, 10)
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
    # plt.tight_layout(pad=5, w_pad=1, h_pad=3.0)
    if not block:
        plt.draw()
        plt.pause(0.5)
    else:
        plt.show()

def get_configured_plot(df):

    subjects = numpy.unique(df[ExperimentFields.subject.value].values)
    experiment = numpy.unique(df[ExperimentFields.experiment.value].values)
    tasks = numpy.unique(df[ExperimentFields.task_id.value].values)
    records = numpy.unique(df[ExperimentFields.record_id.value].values)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
    fig.set_size_inches(17, 10)
    ax1.set_title('Right Hand')
    ax1.set_ylabel('Angle position [deg]')
    ax1.grid()
    ax1.legend([e.value for e in RightHand])
    ax2.set_title('Left Hand')
    ax2.set_ylabel('Angle position [deg]')
    ax2.set_xlabel('Time [s]')
    ax2.grid()
    ax2.legend([e.value for e in LeftHand])

    fig.suptitle("Subject: %s - Experiment: %s \n Record: %s - Task: %s" %
                 (subjects, experiment, records, tasks))
    plt.autoscale(True)

    return fig, ax1, ax2

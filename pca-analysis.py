import numpy
import scipy
import pandas

from sklearn.decomposition import PCA

from utils.visualization import *
from utils.data_loader import *

if __name__ == '__main__':
    DATABASE_PATH = 'kine-adl-be-uji_dataset/CSV Data'

    subjects = EXP1_SUBJECTS
    records = EXP1_RECORDS
    experiment = 1

    df = load_static_grasps(database_path=DATABASE_PATH,
                            subjects_id=subjects,
                            experiment_number=experiment,
                            records_id=records)

    stable_grasps = df[[e.value for e in RightHand] + [e.value for e in LeftHand]]

    # Remove NANs
    data_with_missing_values = stable_grasps.index[stable_grasps.isna().any(axis=1)]
    if len(data_with_missing_values) > 0:
        print("Removing %.2f%% observations with missing numerical values" %
              ((len(data_with_missing_values) / stable_grasps.shape[0]) * 100))
        stable_grasps = stable_grasps.drop(index=data_with_missing_values)


    # Apply PCA
    n_components = 10

    pca = PCA(n_components=n_components)
    pca.fit_transform(stable_grasps)

    # ------------------------------------------------------------------------------------------------
    # PRINT EIGEN VALUES VARIANCES
    labels = ['PC%d' % (c + 1) for c in range(n_components)]

    x = numpy.arange(n_components)  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    # fig.set_size_inches((7, 5))
    # for t in thresholds:
    rects = ax.bar(x, pca.explained_variance_ratio_*100, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of variance explained')
    ax.set_xlabel('Components')
    ax.set_title('PCA on %d stable grasps - KINE_ADL_BE_UJI dataset'
                 '\nsubjects:(%d-%d)-records:(%d-%d)' % (stable_grasps.shape[0],
                                                         min(subjects), max(subjects),
                                                         min(records), max(records)))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid()

    fig.tight_layout()
    plt.savefig('media/pca/E%d/pca-sub_(%d-%d)-records_(%d-%d).png' %
                (experiment, min(subjects), max(subjects),
                 min(records), max(records)))
    plt.show()


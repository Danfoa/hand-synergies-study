import tensorflow as tf
import numpy as np
from regression_model import Regression_Model
from data_loader_tester import DataLoader
from utils.kine_mus_tf_dataset import get_windowed_dataset
import pprint
from utils.data_loader_kine_mus import sEMG, ADLs, load_subject_data, load_subjects_data


if __name__=='__main__':
    # Which joints want to be keept from the data
    DATABASE_PATH = 'kine-mus-uji-dataset/CSV_Data/'

    target_joints = ["WA"]

    epochs = 10
    n_features = 7

    window_sizes = [5, 10, 15, 20]
    shift_periods = [1, 2, 4, 7]
    subjects = [1]

    training_adls = np.random.choice(ADLs, 21, replace=False)
    test_adls = np.array([e for e in ADLs if not e in training_adls])

    for window_size in window_sizes:
        for shift_period in shift_periods:
            # Training dataset
            training_dataset = get_windowed_dataset(dataset_path=DATABASE_PATH,
                                                    window_size=window_size,
                                                    target_joints=target_joints,
                                                    shift_periods=shift_period,
                                                    subjects=subjects,
                                                    adl_ids=training_adls,
                                                    phase_ids=[1, 2, 3])
            # Validation dataset
            test_dataset = get_windowed_dataset(dataset_path=DATABASE_PATH,
                                                window_size=window_size,
                                                target_joints=target_joints,
                                                shift_periods=shift_period,
                                                subjects=subjects,
                                                adl_ids=test_adls,
                                                phase_ids=[1, 2, 3])

            training_dataset = training_dataset.map(lambda d: (d["sEMG"], d["angles_delta"]))
            test_dataset = test_dataset.map(lambda d: (d["sEMG"], d["angles_delta"]))

            # for model, callback, BATCH_SIZE in configurations:
            for configuration in Regression_Model().configurations:
                model, callback, BATCH_SIZE = Regression_Model(window_size, n_features, shift_period).build_conf(*configuration)

                train_dataset = training_dataset.batch(BATCH_SIZE)
                validation_dataset = test_dataset.batch(BATCH_SIZE)
                # testing_dataset = test_dataset.batch(BATCH_SIZE)

                model.summary()
                model.fit(train_dataset,
                  validation_data=validation_dataset,
                  epochs=epochs,
                  callbacks=callback
                  )

            # print("Predicted Value is %s" %model.predict(testing_dataset))
            # print("True Value should be 245")



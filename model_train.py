import tensorflow as tf
import numpy as np
from regression_model import Regression_Model
from data_loader_tester import DataLoader
from utils.kine_mus_tf_dataset import get_windowed_dataset
import pprint
from utils.data_loader_kine_mus import sEMG, ADLs, load_subject_data, load_subjects_data
from logAnglePredCallback import LogAnglePredCallback
import os


if __name__=='__main__':
    # Which joints want to be keept from the data
    DATABASE_PATH = 'kine-mus-uji-dataset/CSV_Data/'

    target_joints = ["WA"]

    epochs = 30
    n_features = 8

    window_sizes = [10, 20, 30]
    subjects = [1]

    training_adls = [17, 18, 11, 5, 12, 23, 2, 25, 21, 7, 20, 14, 15, 19, 8, 6,
                     26, 10, 3, 4, 9, 16, 22]
    test_adls = [1, 13, 24]

    target_size = 5
    logdir = os.path.join("results", "rnn-hyper-param-search")

    for window_size in window_sizes:
        # Training dataset
        training_dataset = get_windowed_dataset(dataset_path=DATABASE_PATH,
                                                target_size=target_size,
                                                window_size=window_size,
                                                target_joints=target_joints,
                                                shift_periods=target_size,
                                                subjects=subjects,
                                                adl_ids=training_adls,
                                                phase_ids=None)
        # Validation dataset
        test_dataset = get_windowed_dataset(dataset_path=DATABASE_PATH,
                                            window_size=window_size,
                                            target_size=target_size,
                                            target_joints=target_joints,
                                            shift_periods=target_size,
                                            subjects=subjects,
                                            adl_ids=test_adls,
                                            phase_ids=None)

        for configuration in Regression_Model().configurations:
            model, callback, BATCH_SIZE = Regression_Model(window_size, n_features, target_size).build_conf(
                *configuration)

            if model is None:
                continue

            hl, dr, rnn, hu, lr, bs, cg = configuration
            ws = window_size
            ts = target_size
            log_angle_prediction = os.path.join(logdir, "predicted_angles",
                                                "rnn=%s-hl=%d-dr=%d-hu=%d-lr=%s-bs=%d-ws-%d-ts=%d-cg=%s"
                                                % (rnn, hl, dr,  hu, lr, bs, ws, ts, cg))

            train_dataset = training_dataset.cache().shuffle(5000).batch(BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)
            validation_dataset = test_dataset.cache().shuffle(5000).batch(BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)

            # testing_dataset = test_dataset.batch(BATCH_SIZE)

            # display(tf.keras.utils.plot_model(model, show_shapes=True, dpi=72,
            #                                   show_layer_names=True,
            #                                   to_file='model.png'))

            # model.summary()
            model.fit(train_dataset,
                      validation_data=validation_dataset,
                      epochs=epochs,
                      callbacks=[*callback, LogAnglePredCallback(validation_dataset, log_angle_prediction)]
                      )



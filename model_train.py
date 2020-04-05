import tensorflow as tf
import numpy as np
from regression_model import Regression_Model
from data_loader_tester import DataLoader
from utils.kine_mus_tf_dataset import get_windowed_dataset
import pprint


if __name__=='__main__':
    # Which joints want to be keept from the data
    DATABASE_PATH = 'kine-mus-uji-dataset/CSV_Data/'

    target_joints = ["WA"]

    epochs = 10
    n_features = 7

    window_sizes = [5, 10, 15, 20]
    dataset = get_windowed_dataset(dataset_path=DATABASE_PATH, window_size=10, target_joints=target_joints,
                                   shift_periods=1, subjects=[1], adl_ids=[1], phase_ids=[1, 3])


    for window_size in window_sizes:
        configurations = Regression_Model(window_size, n_features).configurations
        for model, callback, BATCH_SIZE in configurations:
            train_dataset = dataset.map(lambda angles_delta, angles, sEMG: (angles, sEMG))

            train_dataset = train_dataset.batch(BATCH_SIZE)
            # validation_dataset = val_dataset.batch(BATCH_SIZE)
            # testing_dataset = test_dataset.batch(BATCH_SIZE)

            model.summary()
            model.fit(train_dataset,
              # validation_data=validation_dataset,
              epochs=epochs,
              callbacks=callback
              )

        # print("Predicted Value is %s" %model.predict(testing_dataset))
        # print("True Value should be 245")



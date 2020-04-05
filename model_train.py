import tensorflow as tf
import numpy as np
from regression_model import Regression_Model
from data_loader_tester import DataLoader


if __name__=='__main__':
    n_steps = 3
    epochs = 10
    n_features = 2

    dataloader = DataLoader(n_steps=n_steps, n_features=n_features)
    X = tf.Variable(dataloader.X, name='input_X', dtype=tf.float32)
    y = tf.Variable(dataloader.y, name='label_y', dtype=tf.float32)
    X_val = tf.Variable(dataloader.X_val, name='input_X_val', dtype=tf.float32)
    y_val = tf.Variable(np.asarray([205]), name='label_y_val', dtype=tf.float32)

    x_test = tf.Variable(dataloader.X_test, name='input_X_test', dtype=tf.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

    print("input X is %s" %X)
    print("labels y are %s" %y)
    print("test X is %s" % X_val)
    print("test y are %s" % y_val)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    configurations = Regression_Model().configurations
    for model, callback, BATCH_SIZE in configurations:

        train_dataset = dataset.batch(BATCH_SIZE)
        validation_dataset = val_dataset.batch(BATCH_SIZE)
        testing_dataset = test_dataset.batch(BATCH_SIZE)

        model.summary()
        model.fit(train_dataset,
          validation_data=validation_dataset,
          epochs=epochs,
          callbacks=callback
          )

        print("Predicted Value is %s" %model.predict(testing_dataset))
        print("True Value should be 245")



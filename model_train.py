import tensorflow as tf
import numpy as np
from regression_model import Regression_Model
from data_loader_tester import DataLoader


if __name__=='__main__':
    n_steps = 3
    epochs = 30
    n_features = 2
    BATCH_SIZE = 32

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

    dataset = dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    configurations = Regression_Model().configurations
    for model, callback in configurations:
        model.summary()
        model.fit(dataset,
          validation_data=val_dataset,
          epochs=epochs,
          callbacks=callback
          )

        print("Predicted Value is %s" %model.predict(test_dataset))



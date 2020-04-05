import numpy as np

class DataLoader():

    def __init__(self, n_steps=None, n_features=None):
        # define input sequence
        in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
        out_seq = np.array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
        self.X, self.y = self.get_test_dataset(in_seq1, in_seq2, out_seq)
        # demonstrate prediction
        x_input = np.array([[80, 85], [90, 95], [100, 105]])
        self.X_val = x_input.reshape((1, n_steps, n_features))
        x_test = np.array([[100, 105], [110, 115], [120, 125]])
        self.X_test = x_test.reshape((1, n_steps, n_features))



    # split a multivariate sequence into samples
    def split_sequences(self, sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


    def get_test_dataset(self, in_seq1, in_seq2, out_seq):
        # convert to [rows, columns] structure
        length = len(in_seq1)

        in_seq1 = in_seq1.reshape((length, 1))
        in_seq2 = in_seq2.reshape((length, 1))
        out_seq = out_seq.reshape((length, 1))
        # horizontally stack columns
        dataset = np.hstack((in_seq1, in_seq2, out_seq))

        # choose a number of time steps
        n_steps = 3
        # convert into input/output
        X, y = self.split_sequences(dataset, n_steps)
        return X, y
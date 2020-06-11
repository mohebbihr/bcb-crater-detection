from sklearn.model_selection import train_test_split
import numpy as np

class DataSet(object):
    def __init__(self, images, one_hot, build_images_col=False):
        self._images = images
        self._labels = one_hot
        self._cls = np.argmax(self.labels, axis=1)
        self._index_in_epoch = 0
        self._num_examples = len(images)
        self._epochs_completed = 0
        self._images_col = None
        if build_images_col:
            self._images_col = []
            for img in images:
                self._images_col.append(img.reshape(len(img), 1))
        
    @property
    def images(self):
        return self._images

    @property
    def images_col(self):
        return self._images_col

    @property
    def labels(self):
        return self._labels

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

class Data(object):
    def __init__(self, images, one_hot, random_state=42, build_images_col=False):
        # split data
        X_train, X_test, Y_train, Y_test = \
            train_test_split(images, one_hot, test_size=0.3, random_state=random_state)
        X_validation, X_test, Y_validation, Y_test = \
            train_test_split(X_test, Y_test, test_size=0.5, random_state=random_state)
        
        self.train = DataSet(X_train, Y_train, build_images_col=build_images_col)
        self.validation = DataSet(X_validation, Y_validation, build_images_col=build_images_col)
        self.test = DataSet(X_test, Y_test, build_images_col=build_images_col)

class KCV_Data(object):
    def __init__(self, X_train, X_test, Y_train, Y_test, build_images_col=False):
        
        self.train = DataSet(X_train, Y_train, build_images_col=build_images_col)
        self.test = DataSet(X_test, Y_test, build_images_col=build_images_col)


import tensorflow as tf
from network_blocks import *
from crater_plots import *
import math
import time
from datetime import timedelta
import numpy as np
from sklearn.metrics import f1_score

class Network(object):
    def __init__(self, img_shape=(30, 30, 1), num_classes=2):
        self._img_h, self._img_w = img_shape[:2]
        self._img_shape = img_shape
        self._img_size_flat = self._img_h * self._img_w
        self._num_channels = img_shape[-1]
        self._num_classes = num_classes
        self._layer_conv = []
        self._weights_conv = []
        self._layer_fc = []
        self._session = None
        self._train_batch_size = 64
        self._data = None
        self._declare_placeholders()
        self._total_iterations = 0
        self._train_history = []
        self._test_history = []
        self._vald_history = []

    @property
    def history(self):
        return np.array([self._train_history, self._test_history, self._vald_history])

    @property
    def filters_weights(self):
        return self._session.run(self._weights_conv)

    def get_filters_activations(self, image):
        return self._session.run(self._layer_conv, feed_dict={self._x: [image]})

    def _declare_placeholders(self):
        
        self._x = tf.placeholder(tf.float32, shape=[None, self._img_size_flat], name='x')
        self._x_image = tf.reshape(self._x, [-1, self._img_h, self._img_w, self._num_channels])
        self._y_true = tf.placeholder(tf.float32, shape=[None, self._num_classes], name='y_true')
        self._y_true_cls = tf.argmax(self._y_true, axis=1)

        self._next_input = self._x_image
        self._next_input_size = self._num_channels

    def add_convolutional_layer(self, filter_size, num_filters, use_pooling=True):
        layer_conv, weights_conv = new_conv_layer( input=self._next_input,
                                                   num_input_channels=self._next_input_size,
                                                   filter_size=filter_size,
                                                   num_filters=num_filters,
                                                   use_pooling=use_pooling)

        self._next_input = layer_conv
        self._next_input_size = num_filters
        self._layer_conv.append(layer_conv)
        self._weights_conv.append(weights_conv)

    def add_flat_layer(self):
        layer_flat, num_features = flatten_layer(self._next_input)
        self._next_input = layer_flat
        self._next_input_size = num_features

    def add_fc_layer(self, size, use_relu):
        layer_fc = new_fc_layer(input=self._next_input,
                                num_inputs=self._next_input_size,
                                num_outputs=size,
                                use_relu=use_relu)
        self._next_input = layer_fc
        self._next_input_size = size
        self._layer_fc.append(layer_fc)

    def finish_setup(self):
        final_layer = self._next_input
        self._y_pred = tf.nn.softmax(final_layer)
        self._y_pred_cls = tf.argmax(self._y_pred, axis=1)
        self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_layer, labels=self._y_true)
        self._cost = tf.reduce_mean(self._cross_entropy)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._cost)
        self._correct_prediction = tf.equal(self._y_pred_cls, self._y_true_cls)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))
        self._saver = tf.train.Saver()
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())

    def predict(self, images):
        feed_dict = {self._x: images}
        return self._session.run(self._y_pred, feed_dict=feed_dict)

    def set_data(self, data):
        self._data = data

    def _data_is_available(self):
        if not self._data:
            print(  "There is no data available.\n"
                    "Please use model.set_data(data) to attach it to the model\n"
                    "Data must be of class 'Data', with train, validation and test attributes\n"
                    "and 'next_batch' method\n")
            return False
        return True

    def optimize(self, epochs):
        if not self._data_is_available():
            return

        start_time = time.time()

        train_batch_size = 14
        trainset_size = len(self._data.train.labels)

        num_iterations = int(float(trainset_size)/train_batch_size * epochs) +1
        report_interval = int(math.floor(float(trainset_size)/epochs))

        for i in range(self._total_iterations,
                    self._total_iterations + num_iterations):

            x_batch, y_true_batch = self._data.train.next_batch(train_batch_size)

            feed_dict_train = {self._x: x_batch,
                            self._y_true: y_true_batch}

            self._session.run(self._optimizer, feed_dict=feed_dict_train)

            if (i+1) % 5 == 0:
                feed_dict_train = {self._x: self._data.train.images, self._y_true: self._data.train.labels}
                feed_dict_test = {self._x: self._data.test.images, self._y_true: self._data.test.labels}
                feed_dict_vald = {self._x: self._data.validation.images, self._y_true: self._data.validation.labels}

                train_acc = self._session.run(self._accuracy, feed_dict=feed_dict_train)
                test_acc = self._session.run(self._accuracy, feed_dict=feed_dict_test)
                vald_acc = self._session.run(self._accuracy, feed_dict=feed_dict_vald)

                self._train_history.append(train_acc)
                self._test_history.append(test_acc)
                self._vald_history.append(vald_acc)

                msg = ("Completed epochs: {0:>6}, Training Accuracy: {1:>6.1%}, "
                       "Test Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%}" )

                print(msg.format(self._data.train.epochs_completed, train_acc, test_acc, vald_acc))

        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def optimize_no_valid(self, epochs):
        if not self._data_is_available():
            return
        
        start_time = time.time()
        
        train_batch_size = 14
        trainset_size = len(self._data.train.labels)
        
        num_iterations = int(float(trainset_size)/train_batch_size * epochs) +1
        report_interval = int(math.floor(float(trainset_size)/epochs))
        
        for i in range(self._total_iterations,
                       self._total_iterations + num_iterations):
            
            x_batch, y_true_batch = self._data.train.next_batch(train_batch_size)
            
            feed_dict_train = {self._x: x_batch,
                self._y_true: y_true_batch}
            
            self._session.run(self._optimizer, feed_dict=feed_dict_train)
            
            if (i+1) % 5 == 0:
                feed_dict_train = {self._x: self._data.train.images, self._y_true: self._data.train.labels}
                feed_dict_test = {self._x: self._data.test.images, self._y_true: self._data.test.labels}
                
                train_acc = self._session.run(self._accuracy, feed_dict=feed_dict_train)
                test_acc = self._session.run(self._accuracy, feed_dict=feed_dict_test)
                
                self._train_history.append(train_acc)
                self._test_history.append(test_acc)
                
                msg = ("Completed epochs: {0:>6}, Training Accuracy: {1:>6.1%}, "
                       "Test Accuracy: {2:>6.1%}" )
                       
                print(msg.format(self._data.train.epochs_completed, train_acc, test_acc))
            
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def print_test_accuracy(self, show_example_errors=False, show_confusion_matrix=False):

        if not self._data_is_available():
            return

        test_batch_size = 256
        num_test = len(self._data.test.images)

        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        i = 0
        while i < num_test:
            j = min(i + test_batch_size, num_test)

            images = self._data.test.images[i:j, :]
            labels = self._data.test.labels[i:j, :]

            feed_dict = {self._x: images,
                        self._y_true: labels}

            cls_pred[i:j] = self._session.run(self._y_pred_cls, feed_dict=feed_dict)

            i = j

        cls_true = self._data.test.cls
        correct = (cls_true == cls_pred)

        correct_sum = correct.sum()

        acc = float(correct_sum) / num_test

        f1 = f1_score(cls_true, cls_pred, average='weighted')

        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})  , F1: {3: .1%}"
        print(msg.format(acc, correct_sum, num_test, f1))

        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct, data_test=self._data.test, img_shape=self._img_shape)

        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred, data_test=self._data.test)
                
        return acc,f1
    
    def save(self, filename):
        save_path = self._saver.save(self._session, filename)
        print("Model saved in path: %s" % save_path)
        

    def restore(self, filename):
        try:
            self._saver.restore(self._session, filename)
            print("Model restored.")
        except Exception as ex:
            print("There was a problem (%s). Couldn't restore file: %s"
                  % (type(ex).__name__, filename))

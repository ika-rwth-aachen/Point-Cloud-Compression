#  ==============================================================================
#  MIT License
#  #
#  Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
#  #
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  #
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ==============================================================================

import tensorflow as tf

import numpy as np


class TensorBoard(tf.keras.callbacks.TensorBoard):
    """
    Implements a Tensorboard callback which generates preview images of model input
    and current model prediction. Also stores model loss and metrics.
    """

    def __init__(self, log_dir, dataset, batch_size, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_images = 1
        self.max_outputs = 10
        self.custom_tb_writer = tf.summary.create_file_writer(self.log_dir + '/validation')

    def on_train_batch_end(self, batch, logs=None):
        lr = self.model.optimizer.lr
        iterations = self.model.optimizer.iterations
        with self.custom_tb_writer.as_default():
            tf.summary.scalar('iterations_learning_rate', lr.numpy(), iterations)
        super().on_train_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # get first batch of dataset
        inputs, _ = self.dataset.take(1).get_single_element()

        predictions = self.model(inputs[:self.batch_size, :, :])

        inputs = inputs[:self.num_images, :, :]
        predictions = predictions[:self.num_images, :, :].numpy()

        inputs = tf.image.resize(inputs,
                                 size=[inputs.shape[1] * 3, inputs.shape[2] * 3],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        predictions = tf.image.resize(predictions,
                                      size=[predictions.shape[1] * 3, predictions.shape[2] * 3],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        with self.custom_tb_writer.as_default():
            tf.summary.image('Images/Input Image',
                             inputs,
                             max_outputs=self.max_outputs,
                             step=epoch)
            tf.summary.image('Images/Predicted Image',
                             predictions,
                             max_outputs=self.max_outputs,
                             step=epoch)

        super().on_epoch_end(epoch, logs)

    def on_test_end(self, logs=None):
        super().on_test_end(logs)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.custom_tb_writer.close()


class CosineLearningRateScheduler:
    """Implements a Cosine Learning RateScheduler as tf.keras.Callback"""
    def __init__(self, init_learning_rate,
                 min_learning_rate,
                 min_learning_rate_epoch,
                 max_learning_rate_epoch):
        self.lr0 = init_learning_rate
        self.min_lr = min_learning_rate
        self.min_lr_epoch = min_learning_rate_epoch
        self.max_lr_epoch = max_learning_rate_epoch

    def get_callback(self):
        def lr_schedule(epoch, lr):
            """
            Cosine learning rate schedule proposed by Ilya et al. (https://arxiv.org/pdf/1608.03983.pdf).
            The learning rate starts to decay starting from epoch self.max_lr_epoch, and reaches the minimum
            learning rate at epoch self.min_lr_epoch.
            """
            if epoch >= self.min_lr_epoch:
                lr = self.min_lr
            elif epoch <= self.max_lr_epoch:
                lr = self.lr0
            else:
                lr = self.min_lr + 0.5 * (self.lr0 - self.min_lr) * \
                     (1 + np.cos((epoch - self.max_lr_epoch) / (self.min_lr_epoch - self.max_lr_epoch) * np.pi))
            return lr

        return tf.keras.callbacks.LearningRateScheduler(lr_schedule)

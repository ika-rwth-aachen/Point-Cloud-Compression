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

import os
import tensorflow as tf

import image_utils


class LidarCompressionData:
    """
    Class to create a tf.data.Dataset which loads and preprocesses the
    range images.
    """
    def __init__(self, input_dir, crop_size, batch_size, augment=True):
        self.input_dir = input_dir
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.augment = augment

    def build_dataset(self, shuffle_buffer=1024):
        """
        Creates a tf.data.Dataset object which loads train samples,
        preprocesses them and batches the data.
        :param shuffle_buffer: Buffersize for shuffle operation
        :return: tf.data.Dataset object
        """
        dataset = tf.data.Dataset.list_files(os.path.join(self.input_dir, "*.png"))
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(self.raw_sample_to_cachable_sample, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.sample_to_model_input, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.set_shape)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def set_shape(self, inputs, labels):
        """
        Manually set the shape information for all batches as these information get lost
        after using tf.image.random_crop.
        :param inputs: tf.Tensor as the model input
        :param labels: tf.Tensor as the model label
        :return: Tuple of (inputs, labels)
        """
        inputs.set_shape([self.batch_size, self.crop_size, self.crop_size, 1])
        labels.set_shape([self.batch_size, self.crop_size, self.crop_size, 1])
        return inputs, labels

    def raw_sample_to_cachable_sample(self, input_path):
        """
        Loads a sample from path and return as tf.Tensor
        :param input_path: String with path to sample
        :return: tf.Tensor containing the range image
        """
        input_img = image_utils.load_image_uint16(input_path)
        return input_img

    def sample_to_model_input(self, input_tensor):
        """
        Creates the final model input tensors. Performs random crop with self.crop_size
        on original range representation and returns model input and model label. Note that
        label is a copy of input.
        :param input_tensor: tf.Tensor containing the range image in original shape (H, W, 1)
        :return: Tuple (input, label) as (tf.Tensor, tf.Tensor). Both tensors have a
                 shape of (crop_size, crop_size, 1).
        """
        if self.augment:
            # Pad the image with the left 32 columns during training to ensure
            # each column has the same probability to be cropped.
            input_img = tf.concat([input_tensor, input_tensor[:, :self.crop_size, :]], 1)
            input_img = tf.image.random_crop(input_img, [self.crop_size, self.crop_size, 1])
        else:
            # Used mainly by the validation set. Validation set is already pre-cropped
            input_img = tf.image.random_crop(input_tensor, [self.crop_size, self.crop_size, 1])

        # The label is the input image itself
        return input_img, input_img

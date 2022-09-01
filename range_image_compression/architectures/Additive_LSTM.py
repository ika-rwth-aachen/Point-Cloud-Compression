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
from keras.models import Model
from keras.layers import Layer, Lambda
from keras.layers import Conv2D
from keras.layers import Add, Subtract


class RnnConv(Layer):
    """Convolutional LSTM cell
    See detail in formula (4-6) in paper
    "Full Resolution Image Compression with Recurrent Neural Networks"
    https://arxiv.org/pdf/1608.05148.pdf
    Args:
        name: name of current ConvLSTM layer
        filters: number of filters for each convolutional operation
        strides: strides size
        kernel_size: kernel size of convolutional operation
        hidden_kernel_size: kernel size of convolutional operation for hidden state
    Input:
        inputs: input of the layer
        hidden: hidden state and cell state of the layer
    Output:
        newhidden: updated hidden state of the layer
        newcell: updated cell state of the layer
    """

    def __init__(self, name, filters, strides, kernel_size, hidden_kernel_size):
        super(RnnConv, self).__init__()
        self.filters = filters
        self.strides = strides
        self.conv_i = Conv2D(filters=4 * self.filters,
                             kernel_size=kernel_size,
                             strides=self.strides,
                             padding='same',
                             use_bias=False,
                             name=name + '_i')
        self.conv_h = Conv2D(filters=4 * self.filters,
                             kernel_size=hidden_kernel_size,
                             padding='same',
                             use_bias=False,
                             name=name + '_h')

    def call(self, inputs, hidden):
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv_inputs = self.conv_i(inputs)
        conv_hidden = self.conv_h(hidden[0])
        # all gates are determined by input and hidden layer
        in_gate, f_gate, out_gate, c_gate = tf.split(
            conv_inputs + conv_hidden, 4, axis=-1)  # each gate get the same number of filters
        in_gate = tf.nn.sigmoid(in_gate)  # input/update gate
        f_gate = tf.nn.sigmoid(f_gate)
        out_gate = tf.nn.sigmoid(out_gate)
        c_gate = tf.nn.tanh(c_gate)  # candidate cell, calculated from input
        # forget_gate*old_cell+input_gate(update)*candidate_cell
        newcell = tf.multiply(f_gate, hidden[1]) + tf.multiply(in_gate, c_gate)
        newhidden = tf.multiply(out_gate, tf.nn.tanh(newcell))
        return newhidden, newcell


class EncoderRNN(Layer):
    """
    Encoder layer for one iteration.
    Args:
        bottleneck: bottleneck size of the layer
    Input:
        input: output array from last iteration.
               In the first iteration, it is the normalized image patch
        hidden2, hidden3, hidden4: hidden and cell states of corresponding ConvLSTM layers
        training: boolean, whether the call is in inference mode or training mode
    Output:
        encoded: encoded binary array in each iteration
        hidden2, hidden3, hidden4: hidden and cell states of corresponding ConvLSTM layers
    """
    def __init__(self, bottleneck, name=None):
        super(EncoderRNN, self).__init__(name=name)
        self.bottleneck = bottleneck
        self.Conv_e1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, name='Conv_e1')
        self.RnnConv_e1 = RnnConv("RnnConv_e1", 256, (2, 2), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_e2 = RnnConv("RnnConv_e2", 512, (2, 2), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_e3 = RnnConv("RnnConv_e3", 512, (2, 2), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.Conv_b = Conv2D(self.bottleneck, kernel_size=(1, 1), activation=tf.nn.tanh, use_bias=False, name='b_conv')
        self.Sign = Lambda(lambda x: tf.sign(x), name="sign")

    def call(self, input, hidden2, hidden3, hidden4, training=False):
        # with tf.compat.v1.variable_scope("encoder", reuse=True):
        # input size (32,32,3)
        x = self.Conv_e1(input)
        # x = self.GDN(x)
        # (16,16,64)
        hidden2 = self.RnnConv_e1(x, hidden2)
        x = hidden2[0]
        # (8,8,256)
        hidden3 = self.RnnConv_e2(x, hidden3)
        x = hidden3[0]
        # (4,4,512)
        hidden4 = self.RnnConv_e3(x, hidden4)
        x = hidden4[0]
        # (2,2,512)
        # binarizer
        x = self.Conv_b(x)
        # (2,2,bottleneck)
        # Using randomized quantization during training.
        if training:
            probs = (1 + x) / 2
            dist = tf.compat.v1.distributions.Bernoulli(probs=probs, dtype=input.dtype)
            noise = 2 * dist.sample(name='noise') - 1 - x
            encoded = x + tf.stop_gradient(noise)
        else:
            encoded = self.Sign(x)
        return encoded, hidden2, hidden3, hidden4


class DecoderRNN(Layer):
    """
    Decoder layer for one iteration.
    Input:
        input: decoded array in each iteration
        hidden2, hidden3, hidden4, hidden5: hidden and cell states of corresponding ConvLSTM layers
        training: boolean, whether the call is in inference mode or training mode
    Output:
        decoded: decoded array in each iteration
        hidden2, hidden3, hidden4, hidden5: hidden and cell states of corresponding ConvLSTM layers
    """
    def __init__(self, name=None):
        super(DecoderRNN, self).__init__(name=name)
        self.Conv_d1 = Conv2D(512, kernel_size=(1, 1), use_bias=False, name='d_conv1')
        self.RnnConv_d2 = RnnConv("RnnConv_d2", 512, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_d3 = RnnConv("RnnConv_d3", 512, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_d4 = RnnConv("RnnConv_d4", 256, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_d5 = RnnConv("RnnConv_d5", 128, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.Conv_d6 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', use_bias=False, name='d_conv6',
                              activation=tf.nn.tanh)
        self.DTS1 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_1")
        self.DTS2 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_2")
        self.DTS3 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_3")
        self.DTS4 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_4")
        self.Add = Add(name="add")
        self.Out = Lambda(lambda x: x*0.5, name="out")

    def call(self, input, hidden2, hidden3, hidden4, hidden5, training=False):
        # (2,2,bottleneck)
        x_conv = self.Conv_d1(input)
        # (2,2,512)
        hidden2 = self.RnnConv_d2(x_conv, hidden2)
        x = hidden2[0]
        # (2,2,512)
        x = self.Add([x, x_conv])
        x = self.DTS1(x)
        # (4,4,128)
        hidden3 = self.RnnConv_d3(x, hidden3)
        x = hidden3[0]
        # (4,4,512)
        x = self.DTS2(x)
        # (8,8,128)
        hidden4 = self.RnnConv_d4(x, hidden4)
        x = hidden4[0]
        # (8,8,256)
        x = self.DTS3(x)
        # (16,16,64)
        hidden5 = self.RnnConv_d5(x, hidden5)
        x = hidden5[0]
        # (16,16,128)
        x = self.DTS4(x)
        # (32,32,32)
        # output in range (-0.5,0.5)
        x = self.Conv_d6(x)
        decoded = self.Out(x)
        return decoded, hidden2, hidden3, hidden4, hidden5


class LidarCompressionNetwork(Model):
    """
    The model to compress range image projected from point clouds captured by Velodyne LiDAR sensor
    The encoder and decoder layers are iteratively called for num_iters iterations.
    Details see paper Full Resolution Image Compression with Recurrent Neural Networks
    https://arxiv.org/pdf/1608.05148.pdf. This architecture uses additive reconstruction framework and ConvLSTM layers.
    """
    def __init__(self, bottleneck, num_iters, batch_size, input_size):
        super(LidarCompressionNetwork, self).__init__(name="lidar_compression_network")
        self.bottleneck = bottleneck
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.input_size = input_size

        self.encoder = EncoderRNN(self.bottleneck, name="encoder")
        self.decoder = DecoderRNN(name="decoder")

        self.normalize = Lambda(lambda x: tf.multiply(tf.subtract(x, 0.1), 2.5), name="normalization")
        self.subtract = Subtract()
        self.inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 1))

        self.DIM1 = self.input_size // 2
        self.DIM2 = self.DIM1 // 2
        self.DIM3 = self.DIM2 // 2
        self.DIM4 = self.DIM3 // 2

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.metric_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")

        self.beta = 1.0 / self.num_iters

    def compute_loss(self, res):
        loss = tf.reduce_mean(tf.abs(res))
        return loss

    def initial_hidden(self, batch_size, hidden_size, filters, data_type=tf.dtypes.float32):
        """Initialize hidden and cell states, all zeros"""
        shape = tf.TensorShape([batch_size] + hidden_size + [filters])
        hidden = tf.zeros(shape, dtype=data_type)
        cell = tf.zeros(shape, dtype=data_type)
        return hidden, cell

    def call(self, inputs, training=False):
        # Initialize the hidden states when a new batch comes in
        hidden_e2 = self.initial_hidden(self.batch_size, [8, self.DIM2], 256, inputs.dtype)
        hidden_e3 = self.initial_hidden(self.batch_size, [4, self.DIM3], 512, inputs.dtype)
        hidden_e4 = self.initial_hidden(self.batch_size, [2, self.DIM4], 512, inputs.dtype)
        hidden_d2 = self.initial_hidden(self.batch_size, [2, self.DIM4], 512, inputs.dtype)
        hidden_d3 = self.initial_hidden(self.batch_size, [4, self.DIM3], 512, inputs.dtype)
        hidden_d4 = self.initial_hidden(self.batch_size, [8, self.DIM2], 256, inputs.dtype)
        hidden_d5 = self.initial_hidden(self.batch_size, [16, self.DIM1], 128, inputs.dtype)
        outputs = tf.zeros_like(inputs)

        inputs = self.normalize(inputs)
        res = inputs
        for i in range(self.num_iters):
            code, hidden_e2, hidden_e3, hidden_e4 = \
                self.encoder(res, hidden_e2, hidden_e3, hidden_e4, training=training)

            decoded, hidden_d2, hidden_d3, hidden_d4, hidden_d5 = \
                self.decoder(code, hidden_d2, hidden_d3, hidden_d4, hidden_d5, training=training)

            outputs = tf.add(outputs, decoded)
            # Update res as predicted output in this iteration subtract the original input
            res = self.subtract([outputs, inputs])
            self.add_loss(self.compute_loss(res))
        # Denormalize the tensors
        outputs = tf.clip_by_value(tf.add(tf.multiply(outputs, 0.4), 0.1), 0, 1)
        outputs = tf.cast(outputs, dtype=tf.float32)
        return outputs

    def train_step(self, data):
        inputs, labels = data

        # Run forward pass.
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss = sum(self.losses)*self.beta

        # Run backwards pass.
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update & Compute Metrics
        with tf.name_scope("metrics") as scope:
            self.loss_tracker.update_state(loss)
            self.metric_tracker.update_state(outputs, labels)
            metric_result = self.metric_tracker.result()
            loss_result = self.loss_tracker.result()
        return {'loss': loss_result, 'mae': metric_result}

    def test_step(self, data):
        inputs, labels = data
        # Run forward pass.
        outputs = self(inputs, training=False)
        loss = sum(self.losses)*self.beta
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.metric_tracker.update_state(outputs, labels)
        return {'loss': self.loss_tracker.result(), 'mae': self.metric_tracker.result()}

    def predict_step(self, data):
        inputs, labels = data
        outputs = self(inputs, training=False)
        return outputs

    @property
    def metrics(self):
        return [self.loss_tracker, self.metric_tracker]

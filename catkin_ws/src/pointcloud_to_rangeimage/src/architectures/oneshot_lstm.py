import rospy

from pointcloud_to_rangeimage.msg import RangeImage as RangeImage_msg
from pointcloud_to_rangeimage.msg import RangeImageEncoded as RangeImageEncoded_msg

import cv2
from cv_bridge import CvBridge

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Subtract, Lambda, Layer
from tensorflow.keras.layers import Conv2D, ZeroPadding2D
import tensorflow_compression as tfc
import numpy as np


class RnnConv(Layer):
    """
    Convolutional LSTM cell
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
        name: name of encoder layer
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
        self.Conv_e1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, name='Conv_e1')
        self.GDN = tfc.GDN(alpha_parameter=2, epsilon_parameter=0.5, name="gdn")
        self.RnnConv_e1 = RnnConv("RnnConv_e1", 128, (2, 2), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_e2 = RnnConv("RnnConv_e2", 256, (2, 2), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_e3 = RnnConv("RnnConv_e3", 256, (2, 2), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.Conv_b = Conv2D(bottleneck, kernel_size=(1, 1), activation=tf.nn.tanh, use_bias=False, name='b_conv')
        self.Sign = Lambda(lambda x: tf.sign(x), name="sign")

    def call(self, input, hidden2, hidden3, hidden4, training=False):
        # with tf.compat.v1.variable_scope("encoder", reuse=True):
        # input size (32,32,3)
        x = self.Conv_e1(input)
        x = self.GDN(x)
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
        encoded = self.Sign(x)
        return encoded, hidden2, hidden3, hidden4


class DecoderRNN(Layer):
    """
    Decoder layer for one iteration.
    Args:
        name: name of decoder layer
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
        self.Conv_d1 = Conv2D(256, kernel_size=(1, 1), use_bias=False, name='d_conv1')
        self.iGDN = tfc.GDN(alpha_parameter=2, epsilon_parameter=0.5, inverse=True, name="igdn")
        self.RnnConv_d2 = RnnConv("RnnConv_d2", 256, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_d3 = RnnConv("RnnConv_d3", 256, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_d4 = RnnConv("RnnConv_d4", 128, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.RnnConv_d5 = RnnConv("RnnConv_d5", 64, (1, 1), kernel_size=(3, 3), hidden_kernel_size=(3, 3))
        self.Conv_d6 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', name='d_conv6',
                              use_bias=False, activation=tf.nn.tanh)
        self.DTS1 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_1")
        self.DTS2 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_2")
        self.DTS3 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_3")
        self.DTS4 = Lambda(lambda x: tf.nn.depth_to_space(x, 2), name="dts_4")
        self.Add = Add(name="add")
        self.Out = Lambda(lambda x: x*0.5, name="out")

    def call(self, input, hidden2, hidden3, hidden4, hidden5, training=False):
        # (2,2,bottleneck)
        x = self.Conv_d1(input)
        x_igdn = self.iGDN(x)
        # (2,2,512)
        hidden2 = self.RnnConv_d2(x_igdn, hidden2)
        x = hidden2[0]
        # (2,2,512)
        x = self.Add([x, x_igdn])
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


class EncoderModel(Model):
    """
    Encoder engine consisting of encoder layer and decoder layer.
    Args:
        bottleneck: bottleneck size of the network
        num_iters: number of iterations
    Input:
        input: decoded array in each iteration
        training: boolean, whether the call is in inference mode or training mode
    Output:
        codes: list of compressed binary codewords generated through iterations
    """
    def __init__(self, bottleneck, num_iters):
        super(EncoderModel, self).__init__()
        self.num_iters = num_iters
        self.encoder = EncoderRNN(bottleneck, name="encoder")
        self.decoder = DecoderRNN(name="decoder")
        self.normalize = Lambda(lambda x: tf.subtract(x, 0.5), name="normalization")
        self.subtract = Subtract()
        self.inputs = tf.keras.layers.Input(shape=(32, 1824, 1))
        self.DIM1 = 1824 // 2
        self.DIM2 = self.DIM1 // 2
        self.DIM3 = self.DIM2 // 2
        self.DIM4 = self.DIM3 // 2
        self.beta = 1.0 / self.num_iters

    def initial_hidden(self, batch_size, hidden_size, filters):
        """Initialize hidden and cell states, all zeros"""
        shape = [batch_size] + hidden_size + [filters]
        hidden = tf.zeros(shape)
        cell = tf.zeros(shape)
        return hidden, cell

    def call(self, inputs, training=False):
        # Initialize the hidden states when a new batch comes in
        batch_size = inputs.shape[0]
        hidden_e2 = self.initial_hidden(batch_size, [8, self.DIM2], 128)
        hidden_e3 = self.initial_hidden(batch_size, [4, self.DIM3], 256)
        hidden_e4 = self.initial_hidden(batch_size, [2, self.DIM4], 256)
        hidden_d2 = self.initial_hidden(batch_size, [2, self.DIM4], 256)
        hidden_d3 = self.initial_hidden(batch_size, [4, self.DIM3], 256)
        hidden_d4 = self.initial_hidden(batch_size, [8, self.DIM2], 128)
        hidden_d5 = self.initial_hidden(batch_size, [16, self.DIM1], 64)
        codes = []
        inputs = self.normalize(inputs)
        res = inputs
        for i in range(self.num_iters-1):
            code, hidden_e2, hidden_e3, hidden_e4 = \
                self.encoder(res, hidden_e2, hidden_e3, hidden_e4, training=training)

            decoded, hidden_d2, hidden_d3, hidden_d4, hidden_d5 = \
                self.decoder(code, hidden_d2, hidden_d3, hidden_d4, hidden_d5, training=training)

            # Update res as predicted output in this iteration subtract the original input
            res = self.subtract([decoded, inputs])
            codes.append(code)
        code, hidden_e2, hidden_e3, hidden_e4 = self.encoder(res, hidden_e2, hidden_e3, hidden_e4, training=training)
        codes.append(code)
        return codes

    def predict_step(self, data):
        inputs, labels = data
        codes = self(inputs, training=False)
        return codes


class DecoderModel(Model):
    """
    Decoder engine iteratively calling the decoder layer.
    Args:
        bottleneck: bottleneck size of the network
        num_iters: number of iterations
    Input:
        codes: output of Encoder engine
        training: boolean, whether the call is in inference mode or training mode
    Output:
        output_tensor: decoded image patch
    """
    def __init__(self, bottleneck, num_iters):
        super(DecoderModel, self).__init__()
        self.num_iters = num_iters
        self.DIM1 = 1824 // 2
        self.DIM2 = self.DIM1 // 2
        self.DIM3 = self.DIM2 // 2
        self.DIM4 = self.DIM3 // 2
        self.decoder = DecoderRNN(name="decoder")
        self.subtract = Subtract()
        self.inputs = []
        # inputs of shape [(bs, 2, 114, bottleneck), (bs, 2, 114, bottleneck), (bs, 2, 114, bottleneck), ...]
        for i in range(self.num_iters):
            self.inputs.append(tf.keras.layers.Input(shape=(2, self.DIM4, bottleneck)))

    def initial_hidden(self, batch_size, hidden_size, filters):
        """Initialize hidden and cell states, all zeros"""
        shape = [batch_size] + hidden_size + [filters]
        hidden = tf.zeros(shape)
        cell = tf.zeros(shape)
        return hidden, cell

    def call(self, codes, training=False):
        batch_size = codes[0].shape[0]
        hidden_d2 = self.initial_hidden(batch_size, [2, self.DIM4], 256)
        hidden_d3 = self.initial_hidden(batch_size, [4, self.DIM3], 256)
        hidden_d4 = self.initial_hidden(batch_size, [8, self.DIM2], 128)
        hidden_d5 = self.initial_hidden(batch_size, [16, self.DIM1], 64)
        for i in range(self.num_iters):
            decoded, hidden_d2, hidden_d3, hidden_d4, hidden_d5 = \
                self.decoder(codes[i], hidden_d2, hidden_d3, hidden_d4, hidden_d5, training=training)
        output_tensor = tf.clip_by_value(tf.add(decoded, 0.5), 0, 1)
        return output_tensor

    def predict_step(self, data):
        inputs, labels = data
        outputs = self(inputs, training=False)
        return outputs


class MsgEncoder:
    """
    Subscribe to topic /pointcloud_to_rangeimage_node/msg_out,
    compress range image using RNN image compression model,
    azimuth image using JPEG2000 and intensity image using PNG compression.
    Publish message type RangeImageEncoded to topic /msg_encoded.
    """
    def __init__(self):
        self.pub = rospy.Publisher('msg_encoded', RangeImageEncoded_msg, queue_size=10)
        self.sub = rospy.Subscriber("/pointcloud_to_rangeimage_node/msg_out", RangeImage_msg, self.callback)
        self.bridge = CvBridge()

        bottleneck = rospy.get_param("/rnn_compression/bottleneck")
        num_iters = rospy.get_param("/rnn_compression/num_iters")
        weights_path = rospy.get_param("/rnn_compression/weights_path")

        self.encoder = EncoderModel(bottleneck, num_iters)
        self.encoder(np.zeros((1, 32, 1824, 1)))
        zero_codes = []
        for i in range(num_iters):
            zero_codes.append(np.zeros((1, 2, 114, bottleneck)))
        self.encoder.load_weights(weights_path, by_name=True)
        self.normalize = Lambda(lambda x: tf.multiply(tf.subtract(x, 0.1), 2.5), name="normalization")
        self.inputs = tf.keras.layers.Input(shape=(32, 1824, 1))
        self.num_iters = num_iters

    def parse_img(self, range_image):
        """
        Preprocessing of image array for the RNN image compression model.
        Normalize image array and pad it to be divisible by 32. Details can be found in the thesis.
        """
        range_image = np.reshape(range_image, (1, 32, 1812, 1)) / 65535
        image_vec = np.concatenate((range_image, range_image[:, :, :12, :]), axis=2)
        return image_vec

    def callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)
        try:
            range_image = self.bridge.imgmsg_to_cv2(msg.RangeImage, desired_encoding="mono16")
        except CvBridgeError as e:
            print(e)
        try:
            intensity_map = self.bridge.imgmsg_to_cv2(msg.IntensityMap, desired_encoding="mono8")
        except CvBridgeError as e:
            print(e)
        try:
            azimuth_map = self.bridge.imgmsg_to_cv2(msg.AzimuthMap, desired_encoding="mono16")
        except CvBridgeError as e:
            print(e)

        image_vec = self.parse_img(range_image)
        codes = self.encoder(image_vec)
        codes = (np.stack(codes).astype(np.int8) + 1) // 2
        shape = codes.shape
        codes = np.packbits(codes.reshape(-1))

        msg_encoded = RangeImageEncoded_msg()
        msg_encoded.header = msg.header
        msg_encoded.send_time = msg.send_time
        msg_encoded.code = codes.tostring()
        msg_encoded.shape = [shape[0], shape[1], shape[2], shape[3], shape[4]]
        msg_encoded.NansRow = msg.NansRow
        msg_encoded.NansCol = msg.NansCol

        # Compress azimuth image and intensity image with JPEG 2000 and PNG.
        params_jp2 = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100]
        params_png = [cv2.IMWRITE_PNG_COMPRESSION, 10]
        _, azimuth_map_encoded = cv2.imencode('.jp2', azimuth_map, params_jp2)
        msg_encoded.AzimuthMap = azimuth_map_encoded.tostring()
        _, intensity_map_encoded = cv2.imencode('.png', intensity_map, params_png)
        msg_encoded.IntensityMap = intensity_map_encoded.tostring()
        self.pub.publish(msg_encoded)


class MsgDecoder:
    """
    Subscribe to topic /msg_encoded published by the encoder.
    Decompress the images and pack them in message type RangeImage.
    Publish message to the topic /msg_decoded.
    """
    def __init__(self):
        self.pub = rospy.Publisher("msg_decoded", RangeImage_msg, queue_size=10)
        self.sub = rospy.Subscriber("/msg_encoded", RangeImageEncoded_msg, self.callback)
        self.bridge = CvBridge()

        bottleneck = rospy.get_param("/rnn_compression/bottleneck")
        num_iters = rospy.get_param("/rnn_compression/num_iters")
        weights_path = rospy.get_param("/rnn_compression/weights_path")

        self.decoder = DecoderModel(bottleneck, num_iters)
        zero_codes = []
        for i in range(num_iters):
            zero_codes.append(np.zeros((1, 2, 114, bottleneck)))
        self.decoder(zero_codes)
        self.decoder.load_weights(weights_path, by_name=True)

        self.num_iters = num_iters

    def callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)

        # Unpack compressed binary bitstreams
        shape = (msg.shape[0], msg.shape[1], msg.shape[2], msg.shape[3], msg.shape[4])
        code = np.unpackbits(np.fromstring(msg.code, np.uint8))
        code = code.astype(np.float32)*2-1
        encoded_vec = code.reshape(shape)
        codes = []
        for i in range(encoded_vec.shape[0]):
            codes.append(encoded_vec[i])
        decoded_vec = self.decoder(codes)

        # Decode the images
        range_image_decoded = (decoded_vec[0, :, :1812, 0]*65535).numpy().astype(np.uint16)
        azimuth_map_array = np.fromstring(msg.AzimuthMap, np.uint8)
        azimuth_map_decoded = cv2.imdecode(azimuth_map_array, cv2.IMREAD_UNCHANGED)
        intensity_map_array = np.fromstring(msg.IntensityMap, np.uint8)
        intensity_map_decoded = cv2.imdecode(intensity_map_array, cv2.IMREAD_UNCHANGED)

        # Convert OpenCV image to ROS image.
        try:
            range_image = self.bridge.cv2_to_imgmsg(range_image_decoded, encoding="mono16")
        except CvBridgeError as e:
            print(e)
        try:
            intensity_map = self.bridge.cv2_to_imgmsg(intensity_map_decoded, encoding="mono8")
        except CvBridgeError as e:
            print(e)
        try:
            azimuth_map = self.bridge.cv2_to_imgmsg(azimuth_map_decoded, encoding="mono16")
        except CvBridgeError as e:
            print(e)

        # Pack images in ROS message.
        msg_decoded = RangeImage_msg()
        msg_decoded.header = msg.header
        msg_decoded.send_time = msg.send_time
        msg_decoded.RangeImage = range_image
        msg_decoded.IntensityMap = intensity_map
        msg_decoded.AzimuthMap = azimuth_map
        msg_decoded.NansRow = msg.NansRow
        msg_decoded.NansCol = msg.NansCol

        self.pub.publish(msg_decoded)

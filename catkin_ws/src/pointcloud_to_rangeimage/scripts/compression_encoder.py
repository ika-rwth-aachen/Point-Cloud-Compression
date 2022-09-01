import tensorflow as tf
import rospy
from architectures import additive_lstm, additive_lstm_slim, oneshot_lstm, additive_gru, image_compression

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if rospy.get_param("/rnn_compression/xla"):
    tf.config.optimizer.set_jit("autoclustering")

if rospy.get_param("/rnn_compression/mixed_precision"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def main():
    method = rospy.get_param("/compression_method")
    if method == "image_compression":
        encoder = image_compression.MsgEncoder()
    elif method == "additive_lstm":
        encoder = additive_lstm.MsgEncoder()
    elif method == "additive_lstm_slim":
        encoder = additive_lstm_slim.MsgEncoder()
    elif method == "oneshot_lstm":
        encoder = oneshot_lstm.MsgEncoder()
    elif method == "additive_gru":
        encoder = additive_gru.MsgEncoder()
    else:
        raise NotImplementedError

    rospy.init_node('compression_encoder', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    main()

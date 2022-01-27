import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from architectures import additive_lstm, oneshot_lstm, additive_gru, image_compression
import rospy


def main():
    method = rospy.get_param("/compression_method")
    if method == "image_compression":
        encoder = image_compression.MsgEncoder()
    elif method == "additive_lstm":
        encoder = additive_lstm.MsgEncoder()
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

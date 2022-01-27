import numpy as np

import rospy

from pointcloud_to_rangeimage.msg import RangeImage as RangeImage_msg
from pointcloud_to_rangeimage.msg import RangeImageEncoded as RangeImageEncoded_msg

import cv2
from cv_bridge import CvBridge


class MsgEncoder:
    """
    Subscribe to topic /pointcloud_to_rangeimage_node/msg_out,
    compress range image, azimuth image and intensity image using JPEG2000 or PNG compression.
    Publish message type RangeImageEncoded to topic /msg_encoded.
    """
    def __init__(self):
        self.bridge = CvBridge()

        self.image_compression_method = rospy.get_param("/image_compression/image_compression_method")
        self.show_debug_prints = rospy.get_param("/image_compression/show_debug_prints")

        self.pub = rospy.Publisher('msg_encoded', RangeImageEncoded_msg, queue_size=10)
        self.sub = rospy.Subscriber("/pointcloud_to_rangeimage_node/msg_out", RangeImage_msg, self.callback)
        
        self.int_size_sum = 0
        self.ran_size_sum = 0
        self.azi_size_sum = 0
        self.frame_count = 0

    def callback(self,msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)
        # Convert ROS image to OpenCV image.
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

        msg_encoded = RangeImageEncoded_msg()
        msg_encoded.header = msg.header
        msg_encoded.send_time = msg.send_time
        msg_encoded.NansRow = msg.NansRow
        msg_encoded.NansCol = msg.NansCol

        # Compress the images using OpenCV library and pack compressed data in ROS message.
        if self.image_compression_method == "jpeg":
            params_azimuth_jp2 = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100]
            params_intensity_png = [cv2.IMWRITE_PNG_COMPRESSION, 10]
            params_rangeimage_jp2 = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 50]
            
            _, azimuth_map_encoded = cv2.imencode('.jp2', azimuth_map, params_azimuth_jp2)
            _, intensity_map_encoded = cv2.imencode('.png', intensity_map, params_intensity_png)
            _, range_image_encoded = cv2.imencode('.jp2', range_image, params_rangeimage_jp2)
        
        elif self.image_compression_method == "png":
            params_png = [cv2.IMWRITE_PNG_COMPRESSION, 10]
            
            _, azimuth_map_encoded = cv2.imencode('.png', azimuth_map, params_png)
            _, intensity_map_encoded = cv2.imencode('.png', intensity_map, params_png)
            _, range_image_encoded = cv2.imencode('.png', range_image, params_png)
        else:
            raise NotImplementedError

        msg_encoded.AzimuthMap = azimuth_map_encoded.tostring()
        msg_encoded.IntensityMap = intensity_map_encoded.tostring()
        msg_encoded.RangeImage = range_image_encoded.tostring()

        if self.show_debug_prints:
            self.int_size_sum = self.int_size_sum+intensity_map_encoded.nbytes
            self.azi_size_sum = self.azi_size_sum+azimuth_map_encoded.nbytes
            self.ran_size_sum = self.ran_size_sum+range_image_encoded.nbytes
            self.frame_count = self.frame_count + 1

            print("=============Encoded Image Sizes=============")
            print("Range Image:", range_image_encoded.nbytes)
            print("Intensity Map:", intensity_map_encoded.nbytes)
            print("Azimuth Map:", azimuth_map_encoded.nbytes)
            print("Intensity Average:", self.int_size_sum / self.frame_count / intensity_map.nbytes)
            print("Azimuth Average:", self.azi_size_sum / self.frame_count / azimuth_map.nbytes)
            print("Range Average:", self.ran_size_sum / self.frame_count / range_image.nbytes)
            print("=============================================")
        
        self.pub.publish(msg_encoded)


class MsgDecoder:
    """
    Subscribe to topic /msg_encoded published by the encoder.
    Decompress the images and pack them in message type RangeImage.
    Publish message to the topic /msg_decoded.
    """
    def __init__(self):
        self.bridge = CvBridge()

        self.pub = rospy.Publisher('msg_decoded', RangeImage_msg, queue_size=10)
        self.sub = rospy.Subscriber("/msg_encoded", RangeImageEncoded_msg, self.callback)

    def callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)

        # Decode the images.
        azimuth_map_array = np.fromstring(msg.AzimuthMap, np.uint8)
        azimuth_map_decoded = cv2.imdecode(azimuth_map_array,cv2.IMREAD_UNCHANGED)
        intensity_map_array = np.fromstring(msg.IntensityMap, np.uint8)
        intensity_map_decoded = cv2.imdecode(intensity_map_array,cv2.IMREAD_UNCHANGED)
        range_image_array = np.fromstring(msg.RangeImage, np.uint8)
        range_image_decoded = cv2.imdecode(range_image_array,cv2.IMREAD_UNCHANGED)

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

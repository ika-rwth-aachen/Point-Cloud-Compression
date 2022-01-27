cmake_minimum_required(VERSION 2.8.3)

project(pointcloud_to_rangeimage)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  velodyne_pcl
  cv_bridge
  dynamic_reconfigure
  message_generation
  image_transport
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

catkin_python_setup()

generate_dynamic_reconfigure_options(
    cfg/RangeImage.cfg
)

add_message_files(
  FILES
  RangeImageCompressed.msg
  RangeImage.msg
  RangeImageEncoded.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES
  CATKIN_DEPENDS message_runtime
  DEPENDS PCL
)

catkin_install_python(PROGRAMS 
  scripts/compression_encoder.py 
  scripts/compression_decoder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)


## Declare a cpp executable
add_executable(pointcloud_to_rangeimage_node src/pointcloud_to_rangeimage_node.cpp)
add_dependencies(pointcloud_to_rangeimage_node ${PROJECT_NAME}_gencfg pointcloud_to_rangeimage_generate_messages_cpp)
target_link_libraries(pointcloud_to_rangeimage_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${Boost_LIBRARIES}
)

## Declare a cpp executable
add_executable(rangeimage_to_pointcloud_node src/rangeimage_to_pointcloud_node.cpp)
add_dependencies(rangeimage_to_pointcloud_node ${PROJECT_NAME}_gencfg pointcloud_to_rangeimage_generate_messages_cpp)
target_link_libraries(rangeimage_to_pointcloud_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${Boost_LIBRARIES}
)
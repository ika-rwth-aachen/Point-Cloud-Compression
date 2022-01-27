cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
project(snnrmse)


find_package(catkin REQUIRED COMPONENTS
    dynamic_reconfigure
    nodelet
    roscpp
    roslaunch
    std_msgs
)
find_package(PCL REQUIRED COMPONENTS common io)

generate_dynamic_reconfigure_options(
)

catkin_package(
  CATKIN_DEPENDS pcl_msgs roscpp sensor_msgs std_msgs
  DEPENDS PCL
)

catkin_install_python(PROGRAMS scripts/snnrmse_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

install(TARGETS 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


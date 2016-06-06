cmake_minimum_required(VERSION 2.8.3)
project(semantic_segmentation)

#Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  pcl_ros
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(src/backend)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  LabelIntegratedPointCloud.srv
  LabelIntegratedPointInstanceCloud.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES semantic_segmentation
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs geometry_msgs semantic_map_publisher
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${strands_semantic_segmentation_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(semantic_segmentation
#   src/${PROJECT_NAME}/semantic_segmentation.cpp
# )

## Declare a cpp executable

add_executable(semantic_segmentation_integrate_node src/semantic_segmentation_integrate_node.cpp ${strands_semantic_segmentation_SOURCES})
add_executable(dummy_client src/dummy_client.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes

add_dependencies(semantic_segmentation_integrate_node semantic_segmentation_generate_messages_cpp semantic_map_publisher_generate_messages_cpp)
add_dependencies(dummy_client semantic_segmentation_generate_messages_cpp)


#Moved from the backend cmakelist to make them callable from rosrun.
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
add_definitions(-DLIBF_ENABLE_OPENMP)
#train executable.
add_executable(voxelize ${strands_semantic_segmentation_VOXELIZE_SRCS})
target_link_libraries(voxelize libforest ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})

#train executable.
add_executable(semantic_segmentation_train ${strands_semantic_segmentation_SEM_TRAIN_SRCS})
target_link_libraries(semantic_segmentation_train libforest ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})

#test executable.
add_executable(semantic_segmentation_test ${strands_semantic_segmentation_SEM_TEST_SRCS})
target_link_libraries(semantic_segmentation_test libforest densecrf ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})

#Feature visualization
add_executable(visualize ${strands_semantic_segmentation_VISUALIZE_SRCS})
target_link_libraries(visualize libforest ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})


## Specify libraries to link a library or executable target against

target_link_libraries(semantic_segmentation_integrate_node
  libforest
  densecrf
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(dummy_client
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# Mark executable scripts (Python etc.) for installation
# in contrast to setup.py, you can choose the destination
install(PROGRAMS
  scripts/download_rf.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark executables and/or libraries for installation
install(TARGETS semantic_segmentation_integrate_node dummy_client voxelize semantic_segmentation_train semantic_segmentation_test visualize 
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

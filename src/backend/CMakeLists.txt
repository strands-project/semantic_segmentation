cmake_minimum_required(VERSION 2.8)
project(strands_semantic_segmentation)

#Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")

#find other packages
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(PCL REQUIRED COMPONENTS common features visualization segmentation)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#adding libforest
add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/libforest/)
include_directories(${PROJECT_SOURCE_DIR}/third-party/libforest/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/libforest/lib/fastlog/include)
add_definitions(-DLIBF_ENABLE_OPENMP)

#adding densecrf
add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/densecrf/)
include_directories(${PROJECT_SOURCE_DIR}/third-party/densecrf/include)

#Basic include dirs for this project
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third-party/)

#piecewise list of sources.
set(THIRDPARTY_SRCS ${PROJECT_SOURCE_DIR}/third-party/json/jsoncpp.cpp)
set(UTILS_SRCS  ${PROJECT_SOURCE_DIR}/utils/config.cpp
                ${PROJECT_SOURCE_DIR}/utils/cv_util.cpp
                ${PROJECT_SOURCE_DIR}/utils/data_loader.cpp)


#train executable.
add_executable(voxelize voxelize.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS})
target_link_libraries(voxelize libforest ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})

#train executable.
add_executable(semantic_segmentation_train train.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS})
target_link_libraries(semantic_segmentation_train libforest ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})

#test executable.
add_executable(semantic_segmentation_test test.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS})
target_link_libraries(semantic_segmentation_test libforest densecrf ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})

#rectification of images executable.
add_executable(rectify rectify.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS})
target_link_libraries(rectify  ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})

#Feature visualization
add_executable(visualize visualize_features.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS})
target_link_libraries(visualize libforest ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})


# expose stuff to outside
set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}
                                 ${PROJECT_SOURCE_DIR}/third-party/
                                 ${PROJECT_SOURCE_DIR}/third-party/libforest/include
                                 ${PROJECT_SOURCE_DIR}/third-party/densecrf/include 
                                 ${PROJECT_SOURCE_DIR}/third-party/libforest/lib/fastlog/include PARENT_SCOPE)

set(${PROJECT_NAME}_SOURCES ${THIRDPARTY_SRCS} ${UTILS_SRCS} PARENT_SCOPE)
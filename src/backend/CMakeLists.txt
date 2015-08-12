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


#rectification of images executable.
add_executable(rectify rectify.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS})
target_link_libraries(rectify  ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES})


# expose stuff to outside
set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}
                                 ${PROJECT_SOURCE_DIR}/third-party/
                                 ${PROJECT_SOURCE_DIR}/third-party/libforest/include
                                 ${PROJECT_SOURCE_DIR}/third-party/densecrf/include 
                                 ${PROJECT_SOURCE_DIR}/third-party/libforest/lib/fastlog/include PARENT_SCOPE)

set(${PROJECT_NAME}_SOURCES ${THIRDPARTY_SRCS} ${UTILS_SRCS} PARENT_SCOPE)

set(${PROJECT_NAME}_VISUALIZE_SRCS ${PROJECT_SOURCE_DIR}/visualize_features.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS} PARENT_SCOPE)
set(${PROJECT_NAME}_VOXELIZE_SRCS ${PROJECT_SOURCE_DIR}/voxelize.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS} PARENT_SCOPE)
set(${PROJECT_NAME}_SEM_TRAIN_SRCS ${PROJECT_SOURCE_DIR}/train.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS} PARENT_SCOPE)
set(${PROJECT_NAME}_SEM_TEST_SRCS ${PROJECT_SOURCE_DIR}/test.cpp ${THIRDPARTY_SRCS} ${UTILS_SRCS} PARENT_SCOPE)
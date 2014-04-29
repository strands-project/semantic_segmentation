find_package(Boost REQUIRED)

find_package(Boost COMPONENTS serialization REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.3 REQUIRED)
find_package(OpenCV REQUIRED)

set(UTILS_SRCS configuration.cc dataset.cc data_image.cc static_dataset.cc dynamic_dataset.cc random_source.cc time_stamp.cc console_tree.cc color_coding.cc cv_util.cc string_util.cpp)
set(UTILS_HDRS configuration.hh dataset.hh data_image.hh static_dataset.hh dynamic_dataset.hh random_source.hh time_stamp.hh console_tree.hh color_coding.hh cv_util.hh string_util.h)

include_directories(${PROJECT_SOURCE_DIR} ${QT_INCLUDES}  ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})


file(GLOB DATASET_HDRS datasets/*.h color_coding/*.h)
file(GLOB DATASET_SRCS datasets/*.cpp color_coding/*.cpp)


add_library(utils ${UTILS_HDRS} ${UTILS_SRCS}  ${DATASET_HDRS} ${DATASET_SRCS})
target_link_libraries(utils ${Boost_LIBRARIES} ${EIGEN_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS utils DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


set(GENERATE_PCD_SRCS generate_pcd_from_dataset.cpp)
add_executable(generate_pcd_from_dataset ${GENERATE_PCD_SRCS})
target_link_libraries(generate_pcd_from_dataset utils ${PCL_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS generate_pcd_from_dataset DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
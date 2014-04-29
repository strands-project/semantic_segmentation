set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

find_package(Boost COMPONENTS serialization REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.3 REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(${RGBD_Semantic_Segmentation_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}  ${PCL_INCLUDE_DIRS})

set(FOREST_TRAIN_SRCS randomized_decision_forest_train.cc)
add_executable(randomized_decision_forest_train ${FOREST_TRAIN_SRCS})
target_link_libraries(randomized_decision_forest_train randomized_decision_forest utils ${EIGEN_LIBRARIES} ${TBB_LIBRARIES} ${PCL_LIBRARIES})
install(TARGETS randomized_decision_forest_train DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

set(FOREST_TEST_SRCS randomized_decision_forest_test.cc)
add_executable(randomized_decision_forest_test ${FOREST_TEST_SRCS})
target_link_libraries(randomized_decision_forest_test randomized_decision_forest utils ${EIGEN_LIBRARIES} ${TBB_LIBRARIES} ${PCL_LIBRARIES})
install(TARGETS randomized_decision_forest_test DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)


set(RDFS_HDRS forest.hh tree.hh tree_train.hh split_function_generator.hh datasample.hh)
set(RDFS_SRCS forest.cc tree.cc tree_train.cc)

file(GLOB FEATURE_HDRS features/*.hh)
file(GLOB FEATURE_SRCS features/*.cc)

include_directories(${RGBD_Semantic_Segmentation_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${Boost_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS})

add_library(randomized_decision_forest ${RDFS_SRCS} ${RDFS_HDRS} ${FEATURE_HDRS} ${FEATURE_SRCS})
target_link_libraries(randomized_decision_forest utils ${Boost_LIBRARIES}  ${TBB_LIBRARIES} ${EIGEN_LIBRARIES} ${PCL_LIBRARIES})
install(TARGETS randomized_decision_forest DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)



find_package(Boost REQUIRED)

find_package(Boost COMPONENTS serialization REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(LEAF_PATCH_TEST_SRCS leaf_patch_test.cc)
add_executable(leaf_patch_test ${LEAF_PATCH_TEST_SRCS})
target_link_libraries(leaf_patch_test randomized_decision_forest utils ${EIGEN_LIBRARIES} ${TBB_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS leaf_patch_test DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
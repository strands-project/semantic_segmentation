set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

find_package(Boost COMPONENTS serialization REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)

set(SEMANTIC_LABELING_EVALUATION_SRCS semantic_labeling_evaluation.cc)

include_directories(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} )

add_executable(semantic_labeling_evaluation ${SEMANTIC_LABELING_EVALUATION_SRCS} ${EIGEN_LIBRARIES})
target_link_libraries(semantic_labeling_evaluation utils)
install(TARGETS semantic_labeling_evaluation DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

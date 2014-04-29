set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

find_package(Boost 1.50.0 COMPONENTS serialization REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)


include_directories(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})

set(INFERENCE_HDRS densecrf.hh dense_util.hh fastmath.hh permutohedral.hh sse_defs.hh densecrf_evaluator.hh cluster3d.hh)
set(INFERENCE_SRCS densecrf.cc bipartitedensecrf.cc dense_util.cc permutohedral.cc densecrf_evaluator.cc cluster3d.cc) 

add_library(inference ${INFERENCE_SRCS})
target_link_libraries(inference utils ${EIGEN_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS})
install(TARGETS inference DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

#For a batch execution of the crf.
set(BATCH_DENSE_CRF_SRCS batch_dense_crf.cc)
add_executable(batch_dense_crf ${BATCH_DENSE_CRF_SRCS})
target_link_libraries(batch_dense_crf utils inference ${Boost_LIBRARIES}  ${EIGEN_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS})



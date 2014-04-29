project(RGBD_Semantic_Segmentation)

cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_COMPILER icpc) #compile with -fast

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++0x")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(evaluation)
add_subdirectory(inference)
add_subdirectory(randomized_decision_forest)
add_subdirectory(utils)
add_subdirectory(tests)


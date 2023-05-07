cmake_minimum_required(VERSION 3.9)

project( stereo_estimation LANGUAGES C CXX CUDA )

###############
# packages #
###############

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)

ADD_LIBRARY(cudaLibs src/kernels.cu src/computing.cu)

###############
# executables #
###############

set_source_files_properties(Stereo_Matching PROPERTIES LANGUAGE "CUDA")
add_executable( Stereo_Matching src/main.cu )
target_link_libraries( Stereo_Matching ${OpenCV_LIBS} cudaLibs)
set_target_properties( Stereo_Matching PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set_source_files_properties(pointcloud PROPERTIES LANGUAGE "CUDA")
add_executable( pointcloud src/points.cu )
target_link_libraries( pointcloud ${OpenCV_LIBS} cudaLibs)
set_target_properties( pointcloud PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
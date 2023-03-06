cmake_minimum_required(VERSION 3.1)

project(meanshift 
    VERSION 1.1
    DESCRIPTION "Demonstrate the header-only MeanShift.h library that implements a dense variant of the mean-shift algorithm."
    LANGUAGES CXX)

find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS "OpenMP found!")
endif()

add_executable(meanshift-emblem meanshift-emblem.cpp)
target_compile_options(meanshift-emblem PRIVATE "${OpenMP_CXX_FLAGS}")
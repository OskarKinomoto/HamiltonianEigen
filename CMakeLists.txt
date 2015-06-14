cmake_minimum_required(VERSION 3.2)
project(Kwanty)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")

set(SOURCE_FILES main.cpp)
add_executable(Kwanty ${SOURCE_FILES})
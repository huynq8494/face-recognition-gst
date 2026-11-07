cmake_minimum_required(VERSION 3.28)
project(test14 VERSION 1.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(GST REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.24)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::GST)

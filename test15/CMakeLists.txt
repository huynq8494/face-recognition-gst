cmake_minimum_required(VERSION 3.28)
project(test15 VERSION 1.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.20)
pkg_search_module(GSTAPP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_search_module(GSTVIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)
pkg_search_module(CAIRO REQUIRED IMPORTED_TARGET cairo)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
    PkgConfig::GSTREAMER
    PkgConfig::GSTAPP
    PkgConfig::GSTVIDEO
    PkgConfig::CAIRO)

cmake_minimum_required(VERSION 3.22)
project(test04 VERSION 1.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# Find GStreamer components and create imported targets
pkg_search_module(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.20)
pkg_search_module(GSTAPP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_search_module(GSTVIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

set(ONNXRUNTIME_DIR "/home/huynq/libs/onnxruntime-linux-x64-1.26.0")

add_executable(${PROJECT_NAME} main.cpp)

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    PkgConfig::GSTREAMER 
    PkgConfig::GSTAPP 
    PkgConfig::GSTVIDEO 
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

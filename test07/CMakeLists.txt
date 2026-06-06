cmake_minimum_required(VERSION 3.18)
project(test07 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find GStreamer components and create imported targets
find_package(PkgConfig REQUIRED)
pkg_search_module(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.20)
pkg_search_module(GSTAPP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_search_module(GSTVIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

# ── OpenCV ────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── ONNX Runtime ──────────────────────────────────────────────
set(ONNXRUNTIME_DIR "/home/huynq/libs/onnxruntime-linux-x64-1.26.0")

# ── Sources ───────────────────────────────────────────────────
add_executable(${PROJECT_NAME}
    main.cpp
    scrfd.cpp
    edgeface.cpp
    tracker.cpp
    face_recognition_system.cpp
    gst_pipeline.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    PkgConfig::GSTREAMER 
    PkgConfig::GSTAPP 
    PkgConfig::GSTVIDEO 
    ${OpenCV_LIBS}
     ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

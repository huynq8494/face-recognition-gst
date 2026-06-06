cmake_minimum_required(VERSION 3.18)
project(test06 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

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
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
     ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

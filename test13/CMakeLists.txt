cmake_minimum_required(VERSION 3.10)

project(test13)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(rknpu_scrfd_file rknpu2/scrfd_rknn.cc)

add_subdirectory(/home/huynq/rknn_model_zoo/3rdparty/ 3rdparty.out)
add_subdirectory(/home/huynq/rknn_model_zoo/utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)
find_package(OpenCV REQUIRED)

# ── ONNX Runtime ──────────────────────────────────────────────
set(ONNXRUNTIME_DIR "/home/huynq/libs/onnxruntime-linux-aarch64-1.26.0")

add_executable(${PROJECT_NAME}
    main.cc
    gst_pipeline.cpp
    ${rknpu_scrfd_file}
    scrfd.cpp
    edgeface.cpp
    tracker.cpp
    face_recognition_system.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${ONNXRUNTIME_DIR}/include
    ${GST_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

target_compile_options(${PROJECT_NAME} PRIVATE ${GST_CFLAGS_OTHER})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

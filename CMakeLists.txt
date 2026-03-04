cmake_minimum_required(VERSION 3.10)
project(yolo_onnx_cpp_camera_gpu)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime (GPU)
# set(ONNXRUNTIME_CPU_DIR "/home/ming/Downloads/onnxruntime_cpu")
# include_directories(${ONNXRUNTIME_CPU_DIR}/include)
# link_directories(${ONNXRUNTIME_CPU_DIR}/lib)
set(ONNXRUNTIME_GPU_DIR "/home/ming/Downloads/onnxruntime_gpu")
include_directories(${ONNXRUNTIME_GPU_DIR}/include)
link_directories(${ONNXRUNTIME_GPU_DIR}/lib)

# 头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/preprocessor.cpp
    src/inference.cpp
    src/postprocessor.cpp
    src/visualizer.cpp
    src/detector.cpp
    main.cpp
)

add_executable(yolo_infer ${SOURCES})

target_link_libraries(yolo_infer 
    ${OpenCV_LIBS} 
    onnxruntime
)

set_target_properties(yolo_infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_GPU_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_GPU_DIR}/lib"
)
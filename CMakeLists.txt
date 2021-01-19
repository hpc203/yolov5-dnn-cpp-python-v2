cmake_minimum_required(VERSION 3.17)
project(yolov5-dnn-cpp-python-v2)

set(CMAKE_CXX_STANDARD 14)

add_executable(yolov5-dnn-cpp-python-v2 main_yolo.cpp)
find_package(OpenCV REQUIRED)
target_link_libraries(yolov5-dnn-cpp-python-v2 ${OpenCV_LIBS})

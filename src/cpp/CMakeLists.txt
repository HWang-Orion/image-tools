cmake_minimum_required(VERSION 3.12)
project(img_tool)
set(CMAKE_CXX_STANDARD 20)

find_package (OpenCV 4.0.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(img_tool img_tool.cpp)
target_link_libraries(img_tool ${OpenCV_LIBRARIES})
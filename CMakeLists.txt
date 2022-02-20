cmake_minimum_required(VERSION 3.16)

project(stereo_calib LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Other libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
    "main.cpp"
)

target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

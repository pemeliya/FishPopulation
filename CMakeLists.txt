cmake_minimum_required(VERSION 3.9)

project(FishPopulation)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)

file(GLOB sources "*.h" "*.cpp")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(${PROJECT_NAME} ${sources})

#target_link_libraries(${PROJECT_NAME} openmp)


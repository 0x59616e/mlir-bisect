project(mlir-bisect)
cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS)

message(STATUS "Using LLVM in ${LLVM_DIR}")
message(STATUS "Using MLIR in ${MLIR_DIR}")

include(AddLLVM)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_subdirectory(src)
add_subdirectory(utils)
add_subdirectory(test)

add_custom_target(check
    command ${CMAKE_BINARY_DIR}/utils/mlir-bisect-lit -sv ${CMAKE_BINARY_DIR}/test
    COMMENT "Running test suite ..."
    USES_TERMINAL
)
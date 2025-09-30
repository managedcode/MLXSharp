cmake_minimum_required(VERSION 3.24)
project(mlxsharp C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

if(NOT APPLE)
  set(MLX_BUILD_METAL OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/../extern/mlx ${CMAKE_BINARY_DIR}/extern/mlx EXCLUDE_FROM_ALL)

# Stub implementation for testing and development
add_library(mlxsharp SHARED src/mlxsharp.cpp)

target_link_libraries(mlxsharp PRIVATE mlx)

target_include_directories(mlxsharp PRIVATE ${CMAKE_SOURCE_DIR}/../extern/mlx/mlx)

target_compile_definitions(mlxsharp PRIVATE MLXSHARP_EXPORTS)

install(TARGETS mlxsharp DESTINATION lib)

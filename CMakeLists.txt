cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(ENABLE_MKL OFF)

find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(cudss REQUIRED)
find_package(CUDA REQUIRED)

if(ENABLE_MKL)
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG REQUIRED)
endif()

file(GLOB SOURCES *.cpp)


pybind11_add_module(dxslv ${SOURCES})
target_include_directories(dxslv PRIVATE ${CMAKE_SOURCE_DIR} ${cudss_INCLUDE_DIR})
target_link_libraries(dxslv PRIVATE Eigen::Eigen3 ${CUDA_LIBRARIES})
set_property(TARGET dxslv PROPERTY CUDA_ARCHITECTURES 86)
set_target_properties(dxslv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(dxslv
  PRIVATE
    CUDA::cudart
    cudss
    Eigen3::Eigen
)



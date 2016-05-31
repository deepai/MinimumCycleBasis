cmake_minimum_required(VERSION 3.0)
find_package(CUDA REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_35,code=sm_35 -std=c++11 -DMULTI_THREAD  -DNDEBUG -DVERBOSE -Xcompiler -w
    )

add_compile_options(-O3 -std=c++11 -w)
add_definitions(-DMULTI_THREAD -DNDEBUG -DVERBOSE)

include_directories(
    gpu
    include
    .
    )

FILE(GLOB SOURCES "gpu/*.cuh" "gpu/*.cu" "gpu/*.h" "include/*.cpp" "include/*.h")

cuda_add_executable(
    mcb
    mcb/mcb_cycle_deg_two_remove.cpp
    ${SOURCES}
    )

target_link_libraries(
    mcb
    -lgomp
    )
cmake_minimum_required(VERSION 3.0)
find_package(CUDA REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_35,code=sm_35 -std=c++11 -DMULTI_THREAD -DNDEBUG -DVERBOSE -Xcompiler -w
    )

add_compile_options(-O3 -std=c++11 -w)
add_definitions(-DMULTI_THREAD -DNDEBUG -DVERBOSE)

include_directories(
    gpu
    include
    .
    bicc/include
    )

FILE(GLOB GPU_SOURCES "gpu/*.cuh" "gpu/*.cu" "gpu/*.h")
FILE(GLOB HOST_SOURCES  "include/*.cpp" "include/*.h")
FILE(GLOB BICC_SOURCES "bicc/include/*.h")

add_library(hostlib OBJECT ${HOST_SOURCES})
add_library(bicclib OBJECT ${BICC_SOURCES})

cuda_add_executable(
    mcb_cuda
    mcb/mcb_cycle_deg_two_remove.cpp
    ${GPU_SOURCES}
    $<TARGET_OBJECTS:hostlib>
    )

add_executable(
	mcb_cpu
	mcb/mcb_non_cpu_rd.cpp
	$<TARGET_OBJECTS:hostlib>
)

add_executable(
	mcb_cpu_baseline
	mcb/mcb_cycle.cpp
	$<TARGET_OBJECTS:hostlib>
)

add_executable(
    bicc_decomposition
    bicc/src/bcc_decomposition.cpp
    $<TARGET_OBJECTS:bicclib>
    $<TARGET_OBJECTS:hostlib>
)

add_executable(
    Relabeller
    bicc/Relabeller.cpp
    $<TARGET_OBJECTS:bicclib>
    $<TARGET_OBJECTS:hostlib>
)

target_link_libraries(
    mcb_cuda
    -lgomp
    )

target_link_libraries(
    mcb_cpu
    -lgomp
    )

target_link_libraries(
	mcb_cpu_baseline
    -lgomp
)

target_link_libraries(
    Relabeller
    -lgomp
)

target_link_libraries(
    bicc_decomposition
    -lgomp
)

set_target_properties( mcb_cpu mcb_cuda mcb_cpu_baseline
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/mcb"
)

set_target_properties( bicc_decomposition Relabeller
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bicc"
)

configure_file(filelist.sh file_list.sh COPYONLY)
configure_file(run.sh run.sh COPYONLY)
configure_file(main.sh main.sh COPYONLY)
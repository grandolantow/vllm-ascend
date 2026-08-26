cmake_minimum_required(VERSION 3.16.0)
project(gmsq_kernel)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOC_VERSION "$ENV{SOC_VERSION}" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# gmm_situ_quant is arch35-only (Ascend950PR). Skip silently on other SOCs so
# this subtree stays inert if it ever lands in a multi-SOC build tree.
if(NOT SOC_VERSION MATCHES "950")
    message(STATUS "gmm_situ_quant: skipped (requires Ascend950*, got SOC_VERSION=${SOC_VERSION})")
    return()
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gmm_situ_vcv_dev.cpp
)

ascendc_include_directories(kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/vendor/wqbmm
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ascendc/include/highlevel_api
)

ascendc_compile_definitions(kernels PRIVATE
    -DENABLE_CV_COMM_VIA_SSBUF=1
)

if(DEFINED ENV{GMSQ_SANITIZE})
    ascendc_compile_options(kernels PRIVATE -sanitizer -g)
endif()

add_library(gmm_situ_quant SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/gmm_situ_quant_entries.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/register.cpp
)

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_PATH)
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_NPU_PATH)

target_link_libraries(gmm_situ_quant PRIVATE
    kernels
    torch_npu
    tiling_api
    platform
    m
    dl
)

target_link_directories(gmm_situ_quant PRIVATE
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64
)
target_include_directories(gmm_situ_quant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${TORCH_NPU_PATH}/include
    ${TORCH_NPU_PATH}/include/third_party/acl/inc
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
)

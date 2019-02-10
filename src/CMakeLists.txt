INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/common/inc")
INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/src")
set(dir ${CMAKE_BINARY_DIR}/output/)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dir})
set(CMAKE_BUILD_FILES_DIRECTORY ${dir})
set(CMAKE_BUILD_DIRECTORY ${dir})
SET(EXECUTABLE_OUTPUT_PATH "../output")
INCLUDE(FindCUDA)


SET(CUDA_NVCC_FLAGS
    "--relocatable-device-code=false -gencode arch=compute_53,code=compute_53 -gencode arch=compute_53,code=sm_53")

SET(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
#message(${CMAKE_CXX_FLAGS})

SET(LIBRARY_PATH "${CMAKE_BINARY_DIR}/common/lib/linux/x86_64")

#find_package(GLEW REQUIRED)
#if (GLEW_FOUND)
    link_directories(${LIBRARY_PATH})
    link_libraries(GLEW GL GLU X11 Xi Xmu glut)
#endif()
#message(${GLEW})


CUDA_ADD_EXECUTABLE(NWDEM
    particles.cpp particleSystem.cpp particleSystem_cuda.cu render_particles.cpp shaders.cpp)
TARGET_LINK_LIBRARIES(NWDEM GLEW GL GLU X11 Xi Xmu glut)

#ADD_EXECUTABLE()

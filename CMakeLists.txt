cmake_minimum_required(VERSION 3.8)
project(GF2)

set(CMAKE_CXX_STANDARD 11)

find_package(ALPSCore REQUIRED)
find_package(HDF5 COMPONENTS C HL REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message("setting build type to default, i.e. optimized build")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -DBOOST_DISABLE_ASSERTS")

set(SOURCE_FILES main.cpp DFGF2Solver.cpp DFIntegral.h JobsQueue.cpp SimpleGF2Job.h GF2Solver.h base_integral.h tobe_optimized.cpp)

add_executable(GF2 ${SOURCE_FILES})

target_link_libraries(GF2 ${ALPSCore_LIBRARIES} ${HDF5_HL_LIBRARIES})
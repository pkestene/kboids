# use minimum version required by Kokkos
cmake_minimum_required(VERSION 3.16)

project(kboids
  LANGUAGES CXX)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++14 is for Kokkos
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# External dependency
#

# use Nvidia Thrust for sort algorithms implementation
find_package(Thrust)
if (Thrust_FOUND)
  if(Kokkos_ENABLE_CUDA)
    thrust_create_target(Thrust)
  elseif(Kokkos_ENABLE_OPENMP)
    thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)
  endif()
else()
  message(WARNING "Nvidia Thrust is required (even for OpenMP) to build version1")
endif()

option (USE_FORGE "Enable/disable Forge (GUI library)" OFF)
if (USE_FORGE)
  find_package(OpenGL)
  find_package(Forge)
  if(Forge_FOUND)
    add_compile_definitions(FORGE_ENABLED)
  endif()
endif()

option(USE_LIKWID "Enable LIKWID performance counters")
if(USE_LIKWID)
  find_package(likwid)
endif()

#
# sources
#

# first build kokkos
add_subdirectory(external/kokkos)
add_subdirectory(external/argh)

# build the miniapp
add_subdirectory(src)

##################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                         "${CMAKE_CXX_COMPILER_VERSION} "
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("                 ${CMAKE_CXX_COMPILER}")
message("  Kokkos_ENABLE_OPENMP = ${Kokkos_ENABLE_OPENMP}")
message("  Kokkos_ENABLE_CUDA   = ${Kokkos_ENABLE_CUDA}")
if (Kokkos_ENABLE_CUDA)
  message("  Kokkos_ENABLE_CUDA_LAMBDA    = ${Kokkos_ENABLE_CUDA_LAMBDA}")
  message("  Kokkos_ENABLE_CUDA_CONSTEXPR = ${Kokkos_ENABLE_CUDA_CONSTEXPR}")
  message("  Kokkos_ENABLE_CUDA_UVM       = ${Kokkos_ENABLE_CUDA_UVM}")
  message("  Kokkos CUDA   flags          = ${KOKKOS_CUDA_OPTIONS}")
endif(Kokkos_ENABLE_CUDA)
message("  Kokkos_ENABLE_HWLOC  = ${Kokkos_ENABLE_HWLOC}")

if (Kokkos_ENABLE_CUDA)
  message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
  message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
  message("  CUDA Compile flags : ${CUDA_NVCC_FLAGS}")
endif(Kokkos_ENABLE_CUDA)

message("  Forge library enabled  : ${USE_FORGE}")
message("  Forge library found    : ${Forge_FOUND}")
message("  Forge library version  : ${Forge_VERSION}")
message("  Forge libraries        : ${Forge_LIBRARIES}")
message("  Forge include dirs     : ${Forge_INCLUDE_DIRS}")

message("  Likwid library enabled : ${USE_LIKWID}")
message("  Likwid library found   : ${likwid_FOUND}")
#message("  Likwid library version : ${likwid_VERSION}")
message("  likwid libraries       : ${LIKWID_LIBRARIES}")
message("  likwid include dirs    : ${LIKWID_INCLUDE_DIRS}")
message("  likwid nvidia interface: ${LIKWID_NVIDIA_INTERFACE}")

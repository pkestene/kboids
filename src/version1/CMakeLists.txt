# define how to build and link
add_executable(boids_v1 "")

target_sources(boids_v1
  PRIVATE
  Boids.cpp
  main.cpp)

# add timer
if (Kokkos_ENABLE_OPENMP)
  target_sources(boids_v1
    PRIVATE
    ../time/OpenMPTimer.cpp)
elseif (NOT Kokkos_ENABLE_CUDA)
  target_sources(boids_v1
    PRIVATE
    ../time/SimpleTimer.cpp)
endif()


target_include_directories(boids_v1
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/time)

target_link_libraries(boids_v1
  PRIVATE
  Kokkos::kokkos
  argh
  io::lodepng)

if(USE_LIKWID)
  target_link_libraries(boids_v1 PRIVATE likwid::likwid)
  target_compile_definitions(boids_v1 PRIVATE -DLIKWID_PERFMON)
  target_include_directories(boids_v1 PRIVATE ${LIKWID_INCLUDE_DIR})
endif()

if(USE_FORGE)
  target_sources(boids_v1 PRIVATE
    run_cpu.cpp
    run_gpu.cpp)
  target_link_libraries(boids_v1 PRIVATE Forge::forge)
endif()

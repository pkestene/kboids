# define how to build and link
add_executable(boids_v0 "")

target_sources(boids_v0
  PRIVATE
  Boids.cpp
  run.cpp
  main.cpp)

# add timer
if (Kokkos_ENABLE_OPENMP)
  target_sources(boids_v0
    PRIVATE
    ../time/OpenMPTimer.cpp)
elseif (NOT Kokkos_ENABLE_CUDA)
  target_sources(boids_v0
    PRIVATE
    ../time/SimpleTimer.cpp)
endif()


target_include_directories(boids_v0
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/time)

target_link_libraries(boids_v0
  PRIVATE
  Kokkos::kokkos
  argh
  io::lodepng)

if(USE_LIKWID)
  target_link_libraries(boids_v0 PRIVATE likwid::likwid)
  target_compile_definitions(boids_v0 PRIVATE -DLIKWID_PERFMON)
  target_include_directories(boids_v0 PRIVATE ${LIKWID_INCLUDE_DIR})
endif()

if(USE_FORGE)
  target_link_libraries(boids_v0 PRIVATE Forge::forge)
endif()

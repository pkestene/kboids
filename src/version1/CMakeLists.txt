# define how to build and link
add_executable(boids "")

target_sources(boids
  PRIVATE
  Boids.cpp
  run.cpp
  main.cpp)

# add timer
if (Kokkos_ENABLE_OPENMP)
  target_sources(boids
    PRIVATE
    ../time/OpenMPTimer.cpp)
elseif (NOT Kokkos_ENABLE_CUDA)
  target_sources(boids
    PRIVATE
    ../time/SimpleTimer.cpp)
endif()


target_include_directories(boids
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/time)

target_link_libraries(boids
  PRIVATE
  Kokkos::kokkos
  argh
  io::lodepng)

if(Kokkos_ENABLE_CUDA)
  target_link_libraries(boids
    PRIVATE
    Thrust)
elseif(Kokkos_ENABLE_OPENMP)
  target_link_libraries(boids
    PRIVATE
    ThrustOMP)
endif()

if(USE_LIKWID)
  target_link_libraries(boids PRIVATE likwid::likwid)
  target_compile_definitions(boids PRIVATE -DLIKWID_PERFMON)
  target_include_directories(boids PRIVATE ${LIKWID_INCLUDE_DIR})
endif()

if(USE_FORGE)
  target_link_libraries(boids PRIVATE Forge::forge)
endif()

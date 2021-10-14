add_subdirectory(io)

if (Forge_FOUND)
  add_subdirectory(forge)
endif()

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
    time/OpenMPTimer.cpp)
elseif (NOT Kokkos_ENABLE_CUDA)
  target_sources(boids
    PRIVATE
    time/SimpleTimer.cpp)
endif()


target_include_directories(boids
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(boids
  PRIVATE
  Kokkos::kokkos
  argh
  io::lodepng)

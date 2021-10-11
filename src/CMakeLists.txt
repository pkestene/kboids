add_subdirectory(io)

# define how to build and link
add_executable(boids "")

target_sources(boids
  PRIVATE
  Boids.cpp
  run.cpp
  main.cpp)

target_include_directories(boids
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(boids
  PRIVATE
  Kokkos::kokkos
  io::lodepng)

add_library(lodepng "")
target_sources(lodepng PRIVATE
  lodepng.cpp)

add_library(io::lodepng ALIAS lodepng)


# build an example
add_executable(example_encode_PNG "")
target_sources(example_encode_PNG PRIVATE
  example_encode_PNG.cpp)
target_link_libraries(example_encode_PNG PRIVATE io::lodepng)

add_executable(forge_field "")

target_sources(forge_field
  PRIVATE
  field.cpp)

target_include_directories(forge_field
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(forge_field
  PRIVATE
  Forge::forge)

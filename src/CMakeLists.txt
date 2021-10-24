add_subdirectory(io)

if (Forge_FOUND)
  add_subdirectory(forge)
endif()

add_subdirectory(version0)
add_subdirectory(version1)

if (NOT Kokkos_ENABLE_SYCL)
  add_subdirectory(version2)
endif()

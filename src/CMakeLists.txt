add_subdirectory(io)

if (Forge_FOUND)
  add_subdirectory(forge)
endif()

add_subdirectory(version0)

if (Thrust_FOUND)
  add_subdirectory(version1)
endif()

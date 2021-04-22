
# Create legacy library
add_library(chiopInterface OBJECT chiopInterface.cpp)
# ---

if(HIOP_BUILD_SHARED)
  hiop_add_component(
    SOURCES
    chiopInterface.cpp
    INTERFACE_HEADERS
    hiopInterface.h
    hiopInterface.hpp
    hiopVersion.hpp
    hiop_defs.hpp
    COMPONENT_NAME
    interface
    )
endif()

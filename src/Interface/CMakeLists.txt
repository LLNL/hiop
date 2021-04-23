
if(HIOP_BUILD_SHARED)
  hiop_add_component(
    SOURCES
    chiopInterface.cpp
    INTERFACE_HEADERS
    chiopInterface.hpp
    hiopInterface.h
    hiopInterface.hpp
    hiopVersion.hpp
    hiop_defs.hpp # Pulled in from build directory
    COMPONENT_NAME
    interface
    )
endif()

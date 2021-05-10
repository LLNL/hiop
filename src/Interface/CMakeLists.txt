set(hiopInterface_SRC
  chiopInterface.cpp
  )

set(hiopInterface_INTERFACE_HEADERS
  chiopInterface.hpp
  hiopInterface.h
  hiopInterface.hpp
  hiopVersion.hpp
  )

install(
  FILES ${hiopInterface_INTERFACE_HEADERS}
  DESTINATION include
  )

if(HIOP_BUILD_SHARED)
  add_library(hiopInterface OBJECT ${hiopInterface_SRC})
endif()

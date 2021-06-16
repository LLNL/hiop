if(HIOP_BUILD_SHARED)
 set(hiopInterface_SRC
    hiopInterfacePrimalDecomp.cpp	
    chiopInterface.cpp
  )
else()
 set(hiopInterface_SRC
    hiopInterfacePrimalDecomp.cpp	
  )
endif()

set(hiopInterface_INTERFACE_HEADERS
  chiopInterface.hpp
  hiopInterface.h
  hiopInterface.hpp
  hiopVersion.hpp
  hiop_types.h
  )

install(
  FILES ${hiopInterface_INTERFACE_HEADERS}
  DESTINATION include
  )

add_library(hiopInterface OBJECT ${hiopInterface_SRC})

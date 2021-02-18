set(hiopInterface_SRC
  hiopInterfacePrimalDecomp.cpp	
)

add_library(chiopInterface OBJECT chiopInterface.cpp)

add_library(hiopInterface OBJECT ${hiopInterface_SRC})

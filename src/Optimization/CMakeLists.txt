set(hiopOptimization_SRC
  hiopNlpFormulation.cpp
  hiopFactAcceptor.cpp
  hiopIterate.cpp 
  hiopResidual.cpp 
  hiopFilter.cpp 
  hiopAlgFilterIPM.cpp 
  hiopKKTLinSys.cpp 
  hiopKKTLinSysMDS.cpp 
  hiopHessianLowRank.cpp 
  hiopDualsUpdater.cpp 
  hiopNlpTransforms.cpp
)

if(HIOP_SPARSE)
    set(hiopOptimization_SRC ${hiopOptimization_SRC} hiopKKTLinSysSparse.cpp)
endif()

add_library(hiopOptimization OBJECT ${hiopOptimization_SRC})
target_link_libraries(hiopOptimization PUBLIC hiop_math)



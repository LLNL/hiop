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

set(hiopOptimization_SPARSE_SRC
  hiopKKTLinSysSparse.cpp
  )

set(hiopOptimization_INTERFACE_HEADERS
  hiopAlgFilterIPM.hpp
  hiopDualsUpdater.hpp
  hiopFactAcceptor.hpp
  hiopFilter.hpp
  hiopHessianLowRank.hpp
  hiopIterate.hpp
  hiopKKTLinSys.hpp
  hiopKKTLinSysDense.hpp
  hiopKKTLinSysMDS.hpp
  hiopKKTLinSysSparse.hpp
  hiopLogBarProblem.hpp
  hiopNlpFormulation.hpp
  hiopNlpTransforms.hpp
  hiopPDPerturbation.hpp
  hiopResidual.hpp
  )

if(HIOP_SPARSE)
  list(APPEND hiopOptimization_SRC ${hiopOptimization_SPARSE_SRC})
endif()

install(
  FILES ${hiopOptimization_INTERFACE_HEADERS}
  DESTINATION include
  )

add_library(hiopOptimization OBJECT ${hiopOptimization_SRC})
target_link_libraries(hiopOptimization PRIVATE hiop_tpl)

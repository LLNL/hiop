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

set(hiopOptimization_HEADERS
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

# Create legacy library
add_library(hiopOptimization OBJECT ${hiopOptimization_SRC})
target_link_libraries(hiopOptimization PUBLIC hiop_math)
# ---

hiop_add_component(
  SOURCES ${hiopOptimization_SRC}
  INTERFACE_HEADERS ${hiopOptimization_HEADERS}
  COMPONENT_NAME optimization
  )

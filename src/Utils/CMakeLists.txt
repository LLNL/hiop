add_library(hiopUtils OBJECT hiopLogger.cpp hiopOptions.cpp)
if(HIOP_WITH_KRON_REDUCTION)
  add_library(hiopKronRed OBJECT hiopKronReduction.cpp)
endif(HIOP_WITH_KRON_REDUCTION)

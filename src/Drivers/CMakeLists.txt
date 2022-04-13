# Create examples and tests
add_subdirectory(Dense)
if(HIOP_SPARSE)
  add_subdirectory(Sparse)
endif()
add_subdirectory(MDS)
add_subdirectory(PriDec)
add_subdirectory(IpoptAdapter)

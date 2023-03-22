# Set headers to be installed as part of the hiop interface
set(hiopExecBackends_INTERFACE_HEADERS
  ExecSpace.hpp
  ExecPoliciesRajaCudaImpl.hpp
  ExecPoliciesRajaHipImpl.hpp
  ExecPoliciesRajaOmpImpl.hpp
  MemBackendCppImpl.hpp
  MemBackendCudaImpl.hpp
  MemBackendHipImpl.hpp
  MemBackendUmpireImpl.hpp
)

install(FILES ${hiopExecBackends_INTERFACE_HEADERS} DESTINATION include)

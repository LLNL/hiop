# Set headers to be installed as part of the hiop interface
set(hiopExecBackends_INTERFACE_HEADERS
  ExecSpace.hpp
)

install(FILES ${hiopExecBackends_INTERFACE_HEADERS} DESTINATION include)

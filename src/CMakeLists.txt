add_subdirectory(Interface)
add_subdirectory(Optimization)
add_subdirectory(LinAlg)
add_subdirectory(Utils)

# Interface headers not already picked up by another subdirectory
set(hiop_INTERFACE_HEADERS
  ${CMAKE_BINARY_DIR}/hiop_defs.hpp
  ${CMAKE_BINARY_DIR}/FortranCInterface.hpp
  )

install(
  FILES ${hiop_INTERFACE_HEADERS}
  DESTINATION include
  )

# Create shared and static libraries from sources defined in subdirectories
foreach(libtype shared static)
  string(TOUPPER ${libtype} libtype_upper)
  if(${HIOP_BUILD_${libtype_upper}})
    add_library(hiop_${libtype}
      ${libtype_upper}
      $<TARGET_OBJECTS:hiopLinAlg>
      $<TARGET_OBJECTS:hiopOptimization>
      $<TARGET_OBJECTS:hiopUtils>
      $<$<BOOL:${HIOP_BUILD_SHARED}>:$<TARGET_OBJECTS:hiopInterface>>
      )
    target_link_libraries(hiop_${libtype} PUBLIC hiop_tpl)
    set_target_properties(hiop_${libtype} PROPERTIES OUTPUT_NAME hiop)
    install(TARGETS hiop_${libtype} DESTINATION lib)
  endif()
endforeach()

# Create default library pointed to by HiOp::HiOp
add_library(HiOp::HiOp ALIAS ${hiop_default_library_name})

# Naked _hiop_ target is still used elsewhere in the codebase. This should be
# removed...
add_library(hiop ALIAS ${hiop_default_library_name})

# Test drivers in the form of user applications
add_subdirectory(Drivers)
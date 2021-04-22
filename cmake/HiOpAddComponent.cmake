#[[

@file HiOpAddComponent.cmake

Exposes the macro _hiop_add_component()_ which adds a sublibrary to _libhiop_.

Relies on the following input defined variables:
- HIOP_BUILD_SHARED
- HIOP_BUILD_STATIC

#]]

# Ensure at least one library type is being built!
if((NOT HIOP_BUILD_SHARED) AND (NOT HIOP_BUILD_STATIC))
  message(WARNING "Must build at least one of SHARED or STATIC libraries!")
  message(WARNING "Enabling HIOP_BUILD_STATIC.")
  set(HIOP_BUILD_STATIC ON)
endif()

# This is the default library which HiOp::HiOp will point to. Try to use
# shared by default.
set(hiop_default_library_type "SHARED")
if(NOT HIOP_BUILD_SHARED)
  set(hiop_default_library_type "STATIC")
endif()
string(TOLOWER
  ${hiop_default_library_type}
  hiop_default_library_type_lower
  )
set(hiop_default_library_name
  "hiop_${hiop_default_library_type_lower}"
  )
mark_as_advanced(FORCE
  hiop_default_library_type
  hiop_default_library_type_lower
  )

#[[

@brief Add HiOp component library. All sources from component library will be
  compiled into the final library, and all interface headers will be installed.

@param SOURCES Source files to be used in component.
@param INTERFACE_HEADERS Header files from component that must be installed.
@param LINK_LIBRARIES Libraries that should be linked to final targets. Privacy
  qualifiers provided by caller.

#]]
macro(hiop_add_component)
  set(options "")
  set(oneValueArgs COMPONENT_NAME)
  set(multiValueArgs SOURCES INTERFACE_HEADERS LINK_LIBRARIES)
  cmake_parse_arguments(hiop_add_component "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  
  message(STATUS "Configuring component library hiop_${hiop_add_component_COMPONENT_NAME}")
  add_library(hiop_${hiop_add_component_COMPONENT_NAME}
    OBJECT
    ${hiop_add_component_SOURCES}
    )

  if(hiop_add_component_LINK_LIBRARIES)
    target_link_libraries(hiop_${hiop_add_component_COMPONENT_NAME}
      ${hiop_add_component_LINK_LIBRARIES}
      )
  endif()

  target_link_libraries(hiop_${hiop_add_component_COMPONENT_NAME}
    PRIVATE
    hiop_math
    )

  # Install headers
  if(hiop_add_component_INTERFACE_HEADERS)
    install(FILES ${hiop_add_component_INTERFACE_HEADERS} DESTINATION include)
  endif()
endmacro(hiop_add_component)

#[[

]]
macro(hiop_create_concrete_libraries)

  foreach(libtype shared static)
    string(TOUPPER ${libtype} libtype_upper)
    if(${HIOP_BUILD_${libtype_upper}} AND NOT TARGET hiop_${libtype})
      add_library(hiop_${libtype}
        ${libtype_upper}
        $<TARGET_OBJECTS:hiop_optimization>
        $<TARGET_OBJECTS:hiop_utils>
        $<TARGET_OBJECTS:hiop_linalg>
        $<$<BOOL:${HIOP_BUILD_SHARED}>:$<TARGET_OBJECTS:hiop_interface>>
        )
      add_library(HiOp::${libtype_upper} ALIAS hiop_${libtype})
      target_link_libraries(hiop_${libtype} PUBLIC hiop_math)
      set_target_properties(hiop_${libtype} PROPERTIES OUTPUT_NAME hiop)
      install(TARGETS hiop_${libtype} DESTINATION lib)
    endif()
  endforeach()

  # Create default library pointed to by HiOp::HiOp
  add_library(HiOp::HiOp ALIAS ${hiop_default_library_name})

  # Naked _hiop_ target is still used elsewhere in the codebase
  add_library(hiop ALIAS ${hiop_default_library_name})

endmacro(hiop_create_concrete_libraries)

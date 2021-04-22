#[[

@file HiOpAddComponent.cmake

Exposes the macro _hiop_add_component()_ which adds a sublibrary to _libhiop_.

Relies on the following input defined variables:
- HIOP_BUILD_SHARED
- HIOP_BUILD_STATIC

#]]

#[[

@brief Add HiOp component library. All sources from component library will be
  compiled into the final library, and all interface headers will be installed.

@param SOURCES Source files to be used in component.
@param INTERFACE_HEADERS Header files from component that must be installed.

#]]
macro(hiop_add_component)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCES INTERFACE_HEADERS)
  cmake_parse_arguments(hiop_add_component "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  # We currently build all the sources only once, and reuse these object files
  # for both shared and static libraries. If we add support for building for
  # multiple compute devices (eg CUDA *and* HIP), we will need to create two
  # of these object libraries.
  if(NOT TARGET hiop_obj)
    add_library(hiop_obj OBJECT ${hiop_add_component_SOURCES})
    target_link_libraries(hiop_obj PUBLIC hiop_math)
  else()
    target_sources(hiop_obj PRIVATE ${hiop_add_component_SOURCES})
  endif()

  # Install headers
  install(FILES ${hiop_add_component_INTERFACE_HEADERS} DESTINATION include)
endmacro()


#[[

@brief Creates extra alias libraries and ensures consistence between variabels.

@post At least one of HIOP_BUILD_SHARED and HIOP_BUILD_STATIC is truthy
@pre All component libraries have already been created.

#]]
macro(hiop_create_alias_library)
  # This is the default library which HiOp::HiOp will point to. Try to use
  # shared by default.
  set(hiop_add_component_default_library "hiop_shared")
  if(NOT HIOP_BUILD_SHARED)
    set(hiop_add_component_default_library "hiop_static")
  endif()

  # Ensure at least one library type is being built!
  if((NOT HIOP_BUILD_SHARED) AND (NOT HIOP_BUILD_STATIC))
    message(FATAL_ERROR "Must build at least one of shared or static!")
  endif()

  # Create default library pointed to by HiOp::HiOp
  if(NOT TARGET HiOp::HiOp)
    add_library(HiOp::HiOp ALIAS ${hiop_add_component_default_library})
    # Naked _hiop_ target is still used elsewhere in the codebase
    add_library(hiop ALIAS ${hiop_add_component_default_library})
  endif()
endmacro()

#[[

@brief Creates concrete libraries (eg shared and/or static) from hiop_obj
  target.

@note Takes no parameters.
@pre HIOP_BUILD_<libtype> variables must be set before this macro is called.
  This macro depends on these variables to determine which libraries to create.
@note Creates targets HiOp::SHARED and HiOp::STATIC (if appropriate options are
  set) which allow a user to link against a particular version of libhiop from
  their cmake code (once this code is appropriately exported).
@note Calls hiop_configure_components macro.

]]
macro(hiop_create_concrete_libraries)
  foreach(libtype shared static)
    string(TOUPPER ${libtype} libtype_upper)
    if(${HIOP_BUILD_${libtype_upper}})
      if(NOT TARGET hiop_${libtype})
        add_library(hiop_${libtype} ${libtype_upper} $<TARGET_OBJECTS:hiop_obj>)
        add_library(HiOp::${libtype_upper} ALIAS hiop_${libtype})
        target_link_libraries(hiop_${libtype} PUBLIC hiop_math)
        set_target_properties(hiop_${libtype} PROPERTIES OUTPUT_NAME hiop)
        install(TARGETS hiop_${libtype} DESTINATION lib)
      else()
        target_sources(hiop_${libtype} PRIVATE ${hiop_add_component_SOURCES})
      endif()
    endif()
  endforeach()
endmacro()

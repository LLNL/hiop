find_package(Doxygen)

set(HIOP_DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/doxygen
  CACHE PATH "Absolute path to output directory of ExaGO documentation.")

if(NOT EXISTS ${HIOP_DOXYGEN_OUTPUT_DIRECTORY})
  file(MAKE_DIRECTORY ${HIOP_DOXYGEN_OUTPUT_DIRECTORY})
endif()

if(NOT DOXYGEN_FOUND)
  message(STATUS "Could not find doxygen package... Docs will not be created.")
else()
  message(STATUS "Configuring Doxygen documentation")

  # Some values and paths will be configured using ExaGO cmake variables
  configure_file(
    ${PROJECT_SOURCE_DIR}/doc/doxygen/Doxyfile.in
    ${PROJECT_BINARY_DIR}/Doxyfile
    )

  add_custom_target(
    doc
    COMMAND doxygen ${PROJECT_BINARY_DIR}/Doxyfile
    COMMENT "Generate documentation"
    )

  add_custom_target(
    install_doc
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${PROJECT_BINARY_DIR}/doxygen/html
      ${CMAKE_INSTALL_PREFIX}/doc/html
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${PROJECT_BINARY_DIR}/doxygen/latex
      ${CMAKE_INSTALL_PREFIX}/doc/latex
    COMMENT "Install documentation"
    )
endif()

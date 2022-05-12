find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  message(STATUS "Submodule update")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT
  )
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(
      FATAL_ERROR
        "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules manually"
    )
  endif()
endif()

# For each submodule, we check a file in that submodule we expect to find to
# make sure the submodules were downloaded correctly.
foreach(CHK_FILE "eigen/Eigen/Core")
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/tpl/${CHK_FILE}")
    message(
      FATAL_ERROR
      "It seems required submodule was not downloaded! "
      "Please update submodules manually with the following command "
      "and try again."
      "\n\t$ git submodule update --init --recursive\n"
    )
  endif()
endforeach()

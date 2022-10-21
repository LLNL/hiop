find_package(Git QUIET)
if(GIT_FOUND)
  if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
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
        "'git submodule update --init --recursive' failed with ${GIT_SUBMOD_RESULT}. "
        "Please checkout submodules manually with the folowing command and run cmake again:\n"
        "\t$ git submodule update --init --recursive\n"
        )
    else()
      # For each submodule, we check a file in that submodule we expect to find to
      # make sure the submodules were downloaded correctly.
      foreach(CHK_FILE "eigen/Eigen/Core")
        if(NOT EXISTS "${PROJECT_SOURCE_DIR}/tpl/${CHK_FILE}")
          message(
           FATAL_ERROR
           "It seems that submodule ${PROJECT_SOURCE_DIR}/tpl/${CHK_FILE} was not downloaded! "
           "Please update submodules manually with the following command and try again:\n"
           "\t$ git submodule update --init --recursive\n"
           "Alternatively, use '-DHIOP_USE_EIGEN=ON -DHIOP_EIGEN_DIR=/path/to/eigen/root/dir'."
          )
        endif()
      endforeach()

    endif()
  else() #EXISTS .git
    message(
      WARNING
      "It seems you are not using HiOp under a git repository. To enable EIGEN please use "
      "'-DHIOP_USE_EIGEN=ON -DHIOP_EIGEN_DIR=/path/to/eigen/root/dir'."
      )
    set(HIOP_USE_EIGEN "OFF")
  endif() #EXISTS .git
endif(GIT_FOUND)


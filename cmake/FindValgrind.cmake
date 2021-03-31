#[[

If enabled, looks for `valgrind` executable and enables select tests to run
with valgrind.

Users may set the following variables:

- HIOP_VALGRIND_EXE

]]

set(HIOP_VALGRIND_EXE "" CACHE PATH "Path to valgrind executable for memcheck tests.")

if(HIOP_WITH_VALGRIND_TESTS)
  find_program(VALGRIND
    NAMES "valgrind"
    HINTS ${HIOP_VALGRIND_EXE}
    )
  if(NOT VALGRIND)
    message(STATUS "HIOP_WITH_VALGRIND_TESTS is enabled, but valgrind could "
      "not be found. Disabling valgrind tests.")
    set(HIOP_WITH_VALGRIND_TESTS OFF)
  else(NOT VALGRIND)
    set(HIOP_VALGRIND_CMD "${VALGRIND} --error-exitcode=1"
      CACHE STRING
      "Command used to invoke valgrind by HiOp test suite"
      )
    message(STATUS "Found valgrind: ${VALGRIND}")
  endif(NOT VALGRIND)
endif(HIOP_WITH_VALGRIND_TESTS)

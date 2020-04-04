#ifndef __HIOP_TESTS_COMMON_H
#define __HIOP_TESTS_COMMON_H

#include <limits>
#include <cmath>

namespace hiop::tests
{

class TestBase
{

protected:
    static constexpr double zero      = 0.0;
    static constexpr double half      = 0.5;
    static constexpr double one       = 1.0;
    static constexpr double two       = 2.0;
    static constexpr double eps       = 10*std::numeric_limits<double>::epsilon();
    static constexpr int    SKIP_TEST = -1;

    // must be const pointer and const dest for
    // const string declarations to pass
    // -Wwrite-strings
    static constexpr const char * const  RED       = "\033[1;31m";
    static constexpr const char * const  GREEN     = "\033[1;32m";
    static constexpr const char * const  YELLOW    = "\033[1;33m";
    static constexpr const char * const  CLEAR     = "\033[0m";

protected:
    /// Returns true if two real numbers are equal within tolerance
    [[nodiscard]] constexpr
    bool isEqual(const double a, const double b)
    {
        return (std::abs(a - b) < eps);
    }

    /// Prints error output for each rank
    void printMessage(const int fail, const char* funcname, const int rank)
    {
        if(fail > 0)
        {
            std::cout << RED << "--- FAIL: Test " << funcname << " on rank " << rank << CLEAR << "\n";
        }
        else if (fail == SKIP_TEST)
        {
            if(rank == 0)
            {
                std::cout << YELLOW << "--- SKIP: Test " << funcname << CLEAR << "\n";
            }
        }
        else
        {
            if(rank == 0)
            {
                std::cout << GREEN << "--- PASS: Test " << funcname << CLEAR << "\n";
            }
        }
    }

};

} // namespace hiop::tests

#endif

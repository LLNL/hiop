#ifndef __HIOP_TESTS_COMMON_H
#define __HIOP_TESTS_COMMON_H

#include <limits>
#include <cmath>

using real_type             = double;
using local_ordinal_type    = int;
using global_ordinal_type   = long long;

static constexpr real_type zero = 0.0;
static constexpr real_type quarter = 0.25;
static constexpr real_type half = 0.5;
static constexpr real_type one = 1.0;
static constexpr real_type two = 2.0;
static constexpr real_type three = 3.0;
static constexpr real_type eps =
    10*std::numeric_limits<real_type>::epsilon();
static constexpr int SKIP_TEST = -1;

// must be const pointer and const dest for
// const string declarations to pass
// -Wwrite-strings
static constexpr const char * const  RED       = "\033[1;31m";
static constexpr const char * const  GREEN     = "\033[1;32m";
static constexpr const char * const  YELLOW    = "\033[1;33m";
static constexpr const char * const  CLEAR     = "\033[0m";

namespace hiop::tests
{

class TestBase
{
protected:
    /// Returns true if two real numbers are equal within tolerance
    [[nodiscard]] constexpr
    bool isEqual(const real_type a, const real_type b)
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

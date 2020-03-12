#pragma once

#include <iostream>
#include <cmath>
#include <assert.h>
#include <limits>

namespace hiop::tests {

/**
 * @brief Collection of tests for abstract hiopVector implementations.
 *
 * This class contains implementation of all vector unit tests and abstract
 * interface for testing utility functions, which are specific to vector
 * implementation.
 *
 * @pre All input vectors should be allocated to the same size and have
 * the same partitioning.
 *
 * @post All tests return `true` on all ranks if the test fails on any rank
 * and return `false` otherwise.
 *
 */
class VectorTests
{
public:
    using real_type = double;
    using local_ordinal_type = int;
    using global_ordinal_type = long long;

private:
    static constexpr double eps = 10*std::numeric_limits<double>::epsilon();
    static constexpr double zero = 0.0;
    static constexpr double half = 0.5;
    static constexpr double one  = 1.0;
    static constexpr double two  = 2.0;

public:
    VectorTests(){}
    virtual ~VectorTests(){}

    /// Test get_size() method of hiop vector implementation
    bool vectorGetSize(hiop::hiopVector& x, long long answer, int rank)
    {
        bool fail = (x.get_size() != answer);
        printMessage(fail, __func__, rank);
        return fail;
    }

    /// Test setToConstant method of hiop vector implementation
    bool vectorSetToConstant(hiop::hiopVector& x, int& rank)
    {
        int fail = 0;
        int N = getLocalSize(&x);

        for(int i=0; i<N; ++i)
        {
            setElement(&x, i, zero);
        }

        x.setToConstant(one);

        fail = verifyAnswer(&x, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &x);
    }

    /*
     * this[i] = 0
     */
    bool vectorSetToZero(hiop::hiopVector& v, int& rank)
    {
        v.setToConstant(one);

        v.setToZero();

        int fail = verifyAnswer(&v, zero);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test for function that copies data from this to x.
     */
    bool vectorCopyTo(hiop::hiopVector& v, hiop::hiopVector& to, int rank)
    {
        assert(v.get_size() == to.get_size());
        assert(getLocalSize(&v) == getLocalSize(&to));

        to.setToConstant(one);
        v.setToConstant(two);

        auto todata = getLocalData(&to);
        v.copyTo(todata);

        int fail = verifyAnswer(&to, two);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test for function that copies data from x to this.
     */
    bool vectorCopyFrom(hiop::hiopVector& v, hiop::hiopVector& from, int rank)
    {
        assert(v.get_size() == from.get_size());
        assert(getLocalSize(&v) == getLocalSize(&from));

        from.setToConstant(one);
        v.setToConstant(two);
        v.copyFrom(from);

        int fail = verifyAnswer(&v, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] = (pattern[i] == 0 ? 0 : this[i])
     */
    bool vectorSelectPattern(hiop::hiopVector& v, hiop::hiopVector& ix, int rank)
    {
        const int N = getLocalSize(&v);
        // verify partitioning of test vectors is correct
        assert(v.get_size() == ix.get_size());
        assert(N == getLocalSize(&ix));

        v.setToConstant(two);
        ix.setToConstant(one);
        if (rank== 0)
            setElement(&ix, N - 1, zero);

        v.selectPattern(ix);

        int fail = 0;
        for (int i=0; i<N; ++i)
        {
            double val = getElement(&v, i);
            if ((val != two) && !((rank== 0) && (i == N-1) && (val == zero)))
                fail++;
        }
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] *= alpha
     */
    bool vectorScale(hiop::hiopVector& v, int rank)
    {
        v.setToConstant(half);
        v.scale(two);

        int fail = verifyAnswer(&v, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] *= x[i]
     */
    bool vectorComponentMult(hiop::hiopVector& v, hiop::hiopVector& x, int& rank)
    {
        assert(v.get_size() == x.get_size());
        assert(getLocalSize(&v) == getLocalSize(&x));

        v.setToConstant(two);
        x.setToConstant(half);

        v.componentMult(x);

        int fail = verifyAnswer(&v, one);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] /= x[i]
     */
    bool vectorComponentDiv(hiop::hiopVector& v, hiop::hiopVector& x, int rank)
    {
        assert(v.get_size() == x.get_size());
        assert(getLocalSize(&v) == getLocalSize(&x));

        v.setToConstant(one);
        x.setToConstant(two);

        v.componentDiv(x);

        int fail = verifyAnswer(&v, half);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] = (pattern[i] == 0 ? 0 : this[i]/x[i])
     */
    bool vectorComponentDiv_p_selectPattern(
            hiop::hiopVector& v,
            hiop::hiopVector& x,
            hiop::hiopVector& pattern,
            int rank)
    {
        const int N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(v.get_size() == pattern.get_size());
        assert(N == getLocalSize(&x));
        assert(N == getLocalSize(&pattern));


        v.setToConstant(one);
        x.setToConstant(two);
        pattern.setToConstant(one);
        if (rank== 0)
            setElement(&v, N - 1, zero);

        v.componentDiv_p_selectPattern(x, pattern);

        int fail = 0;
        for (int i=0; i<N; ++i)
        {
            double val = getElement(&v, i);
            if ((val != half) && !((rank== 0) && (i == N-1) && (val == zero)))
                fail++;
        }
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test computing 1-norm ||v||  of vector v
     *                            1
     */
    bool vectorOnenorm(hiop::hiopVector& v, int rank)
    {
        v.setToConstant(-one);
        double actual = v.onenorm();
        double expected = v.get_size();

        int fail = (actual != expected);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test computing 2-norm ||v||  of vector v
     *                            2
     */
    bool vectorTwonorm(hiop::hiopVector& v, int rank)
    {
        v.setToConstant(-one);
        double actual = v.twonorm();
        const double expected = sqrt(v.get_size());

        int fail = !isEqual(expected, actual);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * Test infinity-norm = max(abs(this[i]))
     *                       i
     */
    bool vectorInfnorm(hiop::hiopVector& v, int rank)
    {
        const int N = getLocalSize(&v);
        const double expected = two;

        v.setToConstant(one);
        if (rank== 0)
            setElement(&v, N-1, -two);
        double actual = v.infnorm();

        int fail = (expected != actual);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] += alpha * x[i]
     */
    bool vectorAxpy(hiop::hiopVector& v, hiop::hiopVector& x, int rank)
    {
        const int N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(N == getLocalSize(&x));

        const double alpha = half;
        x.setToConstant(two);
        v.setToConstant(one);

        v.axpy(alpha, x);

        int fail = verifyAnswer(&v, two);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] += alpha * x[i] * z[i]
     */
    bool vectorAxzpy(
            hiop::hiopVector& v,
            hiop::hiopVector& x,
            hiop::hiopVector& z,
            int rank)
    {
        const int N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(N == getLocalSize(&x));

        const double alpha = half;
        x.setToConstant(two);
        z.setToConstant(-one);
        v.setToConstant(one);

        v.axzpy(alpha, x, z);

        int fail = verifyAnswer(&v, zero);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

    /*
     * this[i] += alpha * x[i] / z[i]
     */
    bool vectorAxdzpy(
            hiop::hiopVector& v,
            hiop::hiopVector& x,
            hiop::hiopVector& z,
            int rank)
    {
        const int N = getLocalSize(&v);
        assert(v.get_size() == x.get_size());
        assert(N == getLocalSize(&x));

        const int alpha = two;
        x.setToConstant(-one);
        z.setToConstant(half);
        v.setToConstant(two);

        v.axdzpy(alpha, x, z);

        int fail = verifyAnswer(&v, -two);
        printMessage(fail, __func__, rank);

        return reduceReturn(fail, &v);
    }

protected:
    // Interface to methods specific to vector implementation
    virtual void   setElement(hiop::hiopVector* x, int i, double val) = 0;
    virtual double getElement(const hiop::hiopVector* x, int i) = 0;
    virtual int getLocalSize(const hiop::hiopVector* x) = 0;
    virtual double* getLocalData(hiop::hiopVector* x) = 0;
    virtual int verifyAnswer(hiop::hiopVector* x, double answer) = 0;
    virtual bool reduceReturn(int failures, hiop::hiopVector* x) = 0;

    /// Returns true if two real numbers are equal within tolerance
    bool isEqual(double a, double b)
    {
        return (std::abs(a - b) < eps);
    }

    /// Prints error output for each rank
    void printMessage(int fail, const char* funcname, int rank)
    {
        if(fail != 0)
        {
            std::cout << "--- FAIL: Test " << funcname << " on rank " << rank << "\n";
        }
        else
        {
            if(rank == 0)
            {
                std::cout << "--- PASS: Test " << funcname << "\n";
            }
        }
    }

};

} // namespace hiop::tests


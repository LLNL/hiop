#pragma once

#include <random>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <assert.h>
#include <time.h>

namespace hiop::tests {

/**
 * @brief Collection of tests for abstract hiopVector implementations.
 *
 * This class contains implementation of all vector unit tests and abstract
 * interface for testing utility functions, which are specific to vector
 * implementation.
 *
 */
class VectorTests
{
public:
    VectorTests(){}
    virtual ~VectorTests(){}

    /// Test get_size() method of hiop vector implementation
    int vectorGetSize(hiop::hiopVector& x, long long N, int)
    {
        return x.get_size() == N ? 0 : 1;
    }

    /// Test setToConstant method of hiop vector implementation
    int vectorSetToConstant(hiop::hiopVector& x, int)
    {
        int N = getLocalSize(&x);

        for(int i=0; i<N; ++i)
        {
            setElement(&x, i, 0.0);
        }

        x.setToConstant(1.0);

        for(int i=0; i<N; ++i)
        {
            if (getElement(&x, i) != 1.0)
                return 1;
        }

        return 0;
    }

    int vectorSetToZero(hiop::hiopVector& v)
    {
        int N = v.get_size();
        v.setToZero();

        for(int i=0; i<N; ++i)
        {
            if (getElement(&v, i) != 0.0)
                return 1;
        }

        return 0;
    }

    int vectorCopyTo(hiop::hiopVector& v, hiop::hiopVector& _to)
    {
        assert(v.get_size() == _to.get_size());

        const double C1 = 3.0;
        const double C2 = 2.0;
        int N = v.get_size();

        _to.setToConstant(C1);
        v.setToConstant(C2);

        auto to = getData(&_to);
        v.copyTo(to);

        for (int i=0; i<N; i++)
            if (getElement(&v, i) != C2 || to[i] != C2)
                return 1;

        return 0;
    }

    int vectorCopyFrom(hiop::hiopVector& v, hiop::hiopVector& from)
    {
        assert(v.get_size() == from.get_size());

        const double C1 = 3.0;
        const double C2 = 2.0;
        int N = v.get_size();

        from.setToConstant(C1);
        v.setToConstant(C2);
        v.copyFrom(from);

        for (int i=0; i<N; i++)
            if (getElement(&v, i) != C1 || getElement(&from, i) != C1)
                return 1;

        return 0;
    }

    int vectorSelectPattern(hiop::hiopVector& v, hiop::hiopVector& ix)
    {
        assert(v.get_size() == ix.get_size());

        const int N = v.get_size();
        const int n_rand = 10;
        assert(N > n_rand);
        const std::vector<int> idxs = randoms(n_rand, 0, N);

        const double C = 3.0;
        srand( time(NULL) );

        ix.setToConstant(C);
        v.setToConstant(C);

        for (auto& i : idxs)
            setElement(&ix, i, 0);

        v.selectPattern(ix);

        for (auto& i : idxs)
            if (getElement(&v, i) != 0)
                return 1;

        return 0;
    }

    int vectorScale(hiop::hiopVector& v)
    {
        const int N = v.get_size();
        const double C = 0.5;
        const double alpha = 2.0;
        v.setToConstant(C);
        v.scale(alpha);

        for (int i=0; i<N; i++)
            if (getElement(&v, i) != C*alpha)
                return 1;

        return 0;
    }

    int vectorComponentMult(hiop::hiopVector& v, hiop::hiopVector& other)
    {
        assert(v.get_size() == other.get_size());

        const int N = v.get_size();
        const double C1 = 2.0;
        const double C2 = 3.0;
        v.setToConstant(C1);
        other.setToConstant(C2);

        v.componentMult(other);

        for (int i=0; i<N; i++)
            if (getElement(&v, i) != C1*C2)
                return 1;

        return 0;
    }

    int vectorComponentDiv(hiop::hiopVector& v, hiop::hiopVector& other)
    {
        assert(v.get_size() == other.get_size());

        const int N = v.get_size();
        const double C1 = 2.0;
        const double C2 = 3.0;
        v.setToConstant(C1);
        other.setToConstant(C2);

        v.componentDiv(other);

        for (int i=0; i<N; i++)
            if (getElement(&v, i) != C1/C2)
                return 1;

        return 0;
    }

    int vectorComponentDiv_p_selectPattern(
            hiop::hiopVector& v,
            hiop::hiopVector& component,
            hiop::hiopVector& pattern)
    {
        assert(v.get_size() == component.get_size());
        assert(v.get_size() == pattern.get_size());

        const int N = v.get_size();
        const int n_rand = 10;
        assert(N > n_rand);
        const std::vector<int> idxs = randoms(n_rand, 0, N);

        const double C1 = 2.0;
        const double C2 = 3.0;

        // just has to be nonzero
        pattern.setToConstant(0.0);

        component.setToConstant(C1);
        v.setToConstant(C2);

        for (auto& i : idxs)
            setElement(&pattern, i, 1.0);

        v.componentDiv_p_selectPattern(component, pattern);

        for (int i=0; i<N; i++)
        {
            /* 
             * index is not in idxs, and we should therefore
             * check this == this_prev / C1
             */
            if (std::find(idxs.begin(), idxs.end(), i) != idxs.end())
            {
                if (getElement(&v, i) != (C2/C1))
                    return 1;
            }

            // Otherwise, this == 0
            else
            {
                if (getElement(&v, i) != 0)
                    return 1;
            }
        }

        return 0;
    }

    int vectorOnenorm(hiop::hiopVector& v)
    {
        const int N = v.get_size();
        std::vector<double> testCases = { 0.0, 1.0, 2.0 };

        for (auto& tcase : testCases)
        {
            v.setToConstant(tcase);
            auto actual = v.onenorm();

            double expected = 0;
            for (int i=0; i < N; i++)
                expected += abs(tcase);

            if (expected != actual)
                return 1;
        }

        return 0;
    }

    int vectorTwonorm(hiop::hiopVector& v)
    {
        const int N = v.get_size();
        std::vector<double> testCases = { 0.0, 1.0, 2.0 };

        for (auto& tcase : testCases)
        {
            v.setToConstant(tcase);
            auto actual = v.twonorm();

            double expected = 0;
            for (int i=0; i < N; i++)
                expected += pow(tcase, 2);
            expected = sqrt(expected);

            if (expected != actual)
                return 1;
        }
        return 0;
    }

    int vectorInfnorm(hiop::hiopVector& v)
    {
        const int N = v.get_size();
        auto values = randoms(N, -100, 100);
        int i = 0;
        for (auto& val : values)
            setElement(&v, i++, val);

        double expected = abs(getElement(&v, 0));
        for (int i=1; i<N; i++)
        {
            double aux = abs(getElement(&v, i));
            if (expected < aux)
                expected = aux;
        }

        if (expected != v.infnorm())
            return 1;
                
        return 0;
    }

private:
    auto randoms(const int& length, const int& min, const int& max)
        const -> std::vector<int>
    {
        assert(max > 0);
        srand( time(NULL) );
        std::vector<int> idxs (length);

        int i=0;
        while (i < length)
        {
            int rand_idx = (rand() % (max+min)) - min;
            auto found = std::find(idxs.begin(), idxs.end(), rand_idx);
            if (found == idxs.end())
                idxs[i++] = rand_idx;
        }

        return idxs;
    }

protected:
    virtual void   setElement(hiop::hiopVector* x, int i, double val) = 0;
    virtual double getElement(const hiop::hiopVector* x, int i) = 0;
    virtual double* getData(hiop::hiopVector* x) = 0;
    virtual long long getLocalSize(const hiop::hiopVector* x) = 0;
};

} // namespace hiop::tests

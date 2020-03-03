#pragma once

#include <random>
#include <cstdlib>
#include <algorithm>
#include <iostream>
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
    int vectorGetSize(hiop::hiopVector& x, int N)
    {
        return x.get_size() == N ? 0 : 1;
    }

    /// Test setToConstant method of hiop vector implementation
    int vectorSetToConstant(hiop::hiopVector& x)
    {
        int N = x.get_size();

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

        const int C1 = 3.0f;
        const int C2 = 2.0f;
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

        const int C1 = 3.0f;
        const int C2 = 2.0f;
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
        const int C = 3.0f;
        std::vector<int> randoms;
        srand( time(NULL) );

        int i=0;
        while (i < n_rand)
        {
            int rand_idx = rand() % N;
            auto found = std::find(randoms.begin(), randoms.end(), rand_idx);
            if (found == randoms.end())
            {
                randoms.push_back(rand_idx);
                i += 1;
            }
        }

        ix.setToConstant(C);
        v.setToConstant(C);

        for (auto& i : randoms)
            setElement(&ix, i, 0);

        v.selectPattern(ix);

        for (auto& i : randoms)
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

protected:
    virtual void   setElement(hiop::hiopVector* x, int i, double val) = 0;
    virtual double getElement(const hiop::hiopVector* x, int i) = 0;
    virtual double* getData(hiop::hiopVector* x) = 0;
};

} // namespace hiop::tests

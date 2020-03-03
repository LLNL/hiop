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

    int vectorCopyTo(hiop::hiopVector& v)
    {
        const int C = 3.0f;
        v.setToConstant(C);
        int N = v.get_size();
        auto to = new double[N];

        v.copyTo(to);

        int _r = 0;
        for (int i=0; i<N; i++)
            if (getElement(&v, i) != C || to[i] != C)
                _r = 1;

        delete[] to;
        return _r;
    }

    int vectorCopyFrom(hiop::hiopVector& v)
    {
        const int C1 = 3.0f;
        const int C2 = 2.0f;
        int N = v.get_size();

        // TODO: test with other implementations of hiopVector,
        // such that we do not miss any implementation specific
        // errors by only testing copying from a hiopVectorPar
        hiop::hiopVectorPar from(N);
        from.setToConstant(C1);

        v.setToConstant(C2);
        v.copyFrom(from);

        for (int i=0; i<N; i++)
            if (getElement(&v, i) != C1 || getElement(&from, i) != C1)
                return 1;

        return 0;
    }

    int vectorSelectPattern(hiop::hiopVector& v)
    {
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

        hiop::hiopVectorPar ix(N);
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

protected:
    virtual void   setElement(hiop::hiopVector* x, int i, double val) = 0;
    virtual double getElement(const hiop::hiopVector* x, int i) = 0;

};

} // namespace hiop::tests

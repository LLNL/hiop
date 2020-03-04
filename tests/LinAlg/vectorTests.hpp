#pragma once


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


protected:
    virtual void   setElement(hiop::hiopVector* x, int i, double val) = 0;
    virtual double getElement(const hiop::hiopVector* x, int i) = 0;
    virtual long long getLocalSize(const hiop::hiopVector* x) = 0;
};

} // namespace hiop::tests

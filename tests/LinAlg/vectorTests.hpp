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
        int N = v.get_size();
        auto to = new double[N];

        v.copyTo(to);

        int ret_code = 0;

        for (int i=0; i<N; i++)
            if (getElement(&v, i) != to[i])
            {
                ret_code = 1;
                break;
            }

        delete[] to;
        return ret_code;
    }

    int vectorCopyFrom(hiop::hiopVector& v)
    {
        // TODO: test with other implementations of hiopVector,
        // such that we do not miss any implementation specific
        // errors by only testing copying from a hiopVectorPar
        hiop::hiopVectorPar from(v.get_size());

        v.copyFrom(from);

        for (int i=0; i<v.get_size(); i++)
            if (getElement(&v, i) != getElement(&from, i))
                return 1;

        return 0;
    }

protected:
    virtual void   setElement(hiop::hiopVector* x, int i, double val) = 0;
    virtual double getElement(const hiop::hiopVector* x, int i) = 0;

};

} // namespace hiop::tests

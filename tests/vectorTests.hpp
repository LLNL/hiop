#pragma once


namespace hiopTest {

class VectorTests
{
public:
    VectorTests(){}
    virtual ~VectorTests(){}

    int testGetSize(hiop::hiopVector& x, int N)
    {
        return x.get_size() == N ? 0 : 1;
    }

protected:
    virtual void   setElement(hiop::hiopVector* x, int i, double val) = 0;
    virtual double getElement(const hiop::hiopVector* x, int i) = 0;

};

} // namespace hiopTest

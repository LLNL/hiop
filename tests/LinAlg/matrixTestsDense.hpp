#pragma once

#include "matrixTests.hpp"

namespace hiop::tests {

class MatrixTestsDense : public MatrixTests
{
public:
    MatrixTestsDense() {}
    virtual ~MatrixTestsDense(){}

private:
    virtual void setLocalElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j, real_type val);
    virtual real_type getLocalElement(const hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j);
    virtual real_type getLocalElement(const hiop::hiopVector* x, local_ordinal_type i);
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a);
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a);
    virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x);
    virtual int verifyAnswer(hiop::hiopMatrix* A, real_type answer);
    virtual int verifyAnswer(
            hiop::hiopMatrix* A,
            std::function<real_type(local_ordinal_type, local_ordinal_type)> expect);
    virtual int verifyAnswer(hiop::hiopVector* x, real_type answer);
    virtual int verifyAnswer(
            hiop::hiopVector* x,
            std::function<real_type(local_ordinal_type)> expect);
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A);
    virtual bool globalToLocalMap(
            hiop::hiopMatrix* A,
            const global_ordinal_type row,
            const global_ordinal_type col,
            local_ordinal_type& local_row,
            local_ordinal_type& local_col);

#ifdef HIOP_USE_MPI
    MPI_Comm getMPIComm(hiop::hiopMatrix* A);
#endif
};

} // namespace hiop::tests

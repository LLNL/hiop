#include <hiopMatrix.hpp>
#include "matrixTestsDense.hpp"

namespace hiop::tests {

/// Method to set matrix _A_ element (i,j) to _val_.
/// First need to retrieve hiopMatrixDense from the abstract interface
void MatrixTestsDense::setElement(hiop::hiopMatrix* A, int i, int j, double val)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    double** data = amat->get_M();
    data[i][j] = val;
}

/// Returns element (i,j) of matrix _A_.
/// First need to retrieve hiopMatrixDense from the abstract interface
double MatrixTestsDense::getElement(hiop::hiopMatrix* A, int i, int j)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->local_data()[i][j];
}

int MatrixTestsDense::getNumLocRows(hiop::hiopMatrix* A)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->get_local_size_m();
    //                         ^^^
}

int MatrixTestsDense::getNumLocCols(hiop::hiopMatrix* A)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->get_local_size_n();
    //                         ^^^
}

} // namespace hiop::tests

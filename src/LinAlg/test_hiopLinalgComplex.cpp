#include "hiopMatrixComplexSparseTriplet.hpp"
#include "hiopMatrixComplexDense.hpp"

#include <iostream>

using namespace hiop;

int main()
{
  { //TEST sparse complex matrix
    //this is the rectangular matrix to test with
    int m=3, n=5;
    int Mrow[] = {0, 0, 1, 1, 2, 2, 2};
    int Mcol[] = {0, 3, 2, 4, 0, 3, 4};
    std::complex<double> Mval[] = {{1,1}, {1,-1}, {0,1}, {1,0}, {2,-1}, {1.5,-0.5}, {0.5, -0.5}};
    
    int nnz = sizeof(Mrow) / sizeof(Mrow[0]);
    assert(nnz == sizeof Mcol / sizeof Mcol[0]);
    assert(nnz == sizeof Mval / sizeof Mval[0]);
    
    hiopMatrixComplexSparseTriplet mat(m,n,nnz);
    mat.copyFrom(Mrow, Mcol, Mval);
    
    //test1
    double abs_nrm = mat.max_abs_value();
    double diff = std::fabs(abs_nrm-2.23606797749979);
    if(diff>1e-12)
      printf("error: max_abs_value did not return the correct value. Difference: %6.3e\n", diff);
    
    mat.print();
    
    mat.storage()->sort_indexes();
    mat.print();
    
    mat.storage()->sum_up_duplicates();
    mat.print();
  }

  { //TEST dense complex matrix
    hiopMatrixComplexDense mat(3,4);
    std::complex<double>** M = mat.get_M();
    for(int i=0; i<mat.m(); i++)
      for(int j=0; j<mat.n(); j++)
	M[i][j] = std::complex<double>(i,j);

    hiopMatrixComplexDense mat2(3,4);
    std::complex<double>** M2 = mat2.get_M();
    for(int i=0; i<mat2.m(); i++)
      for(int j=0; j<mat2.n(); j++)
	M2[i][j] = std::complex<double>(i,-2.*j);

    mat.print();
    mat2.print();
    mat2.addMatrix(std::complex<double>(1.,0), mat);

    mat2.print();

    //test1
    double abs_nrm = mat.max_abs_value();
    double diff = std::fabs(abs_nrm-3.605551275463989);
    if(diff>1e-12)
      printf("error: max_abs_value did not return the correct value. Difference: %6.3e\n", diff);

  }
}

#include "hiopVector.hpp"
#include "hiopMatrix.hpp"

#ifdef WITH_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#endif

#include <cstdio>
#include <cmath>
#include <cassert>

bool testMatMatProds(long long* col_partion);

int main(int argc, char **argv)
{
  int rank=0, numRanks=1;
#ifdef WITH_MPI
  MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS==MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  assert(MPI_SUCCESS==MPI_Comm_size(MPI_COMM_WORLD,&numRanks));
  if(0==rank) printf("MPI is enabled\n");
#endif
  //decide here the distribution
  long long n = 1024;
  long long* column_part = new long long[numRanks];
  for(int i=0; i<=numRanks; i++) column_part[i]=n/numRanks*i;

  hiopVectorPar v(n, column_part, MPI_COMM_WORLD);
  

  v.setToConstant(0.1);
  double nrm = v.twonorm();
  if(0==rank) printf("Norm is: %g\n", nrm);

  v.scale(2.);
  nrm = v.twonorm();
  if(0==rank) printf("After scalarMult norm is: %g\n", nrm);

  hiopVectorPar* v2 = v.alloc_clone();
  v2->setToConstant(0.1);
  v2->axpy(1.5, v); //v2=v2+1.5*v
  nrm = v2->twonorm();
  if(0==rank) printf("After axpy norm is: %.3f\n", nrm);

  hiopVectorPar* v3 = v2->alloc_clone(); v3->setToConstant(0.1);
  delete v3, v2;

  long long m=2000;
  hiopMatrixDense A(m, n, column_part, MPI_COMM_WORLD);

  double t=MPI_Wtime();
  A.setToConstant(1./n); 
  t = (MPI_Wtime()-t);
  printf("setToConstant time=%g sec\n", t);

  if(rank==0) {A.get_M()[0][1]=2.; A.get_M()[0][2]=3.;} 
  A.print(4, 10, 1);
  hiopVectorPar y(m);
  y.setToConstant(2.);
  v.setToConstant(1.);

  t=MPI_Wtime();
  // y = beta*y + alpha*A*v
  A.timesVec(0., y, 1.0, v);
  if(0==rank) { 
    printf("timesVec time=%g sec\n", MPI_Wtime()-t);
    y.print(stdout, "y",10,0);
  }

  //v = beta*v + alpha*A'*y
  v.setToConstant(1.); y.setToConstant(1.);
  t=MPI_Wtime();
  A.transTimesVec(0.5, v, 1.0, y);
  printf("transTimesVec time=%g sec\n", MPI_Wtime()-t);
  v.print(stdout, "v",10,0);

  //more testing 
  testMatMatProds(column_part);

  delete[] column_part;
#ifdef WITH_MPI
  MPI_Finalize();
#endif
  return 0;
}

bool testMatMatProds(long long* col_partion)
{
  int m,n,k;
  //mat times mat C = beta*C + alphaA*B
  {
    m=2; n=5; k=3;
    double c[]={0.0,0.0,0.0, 6,12,24};
    hiopMatrixDense C(m,k); 
    C.copyFrom(c);

    double a[]={1,2,3,4,5, 0.5,0.5,0.5,0.5,0.5};
    hiopMatrixDense A(m,n);
    A.copyFrom(a);
    
    double b[]={1,4,-1, 2,0,-1, 1,0,0, 2,0,0, 1,0,0};
    hiopMatrixDense B(n,k);
    B.copyFrom(b);

    A.timesMat(0.5, C, 2.0, B);
    C.print();
    
    double true_result[]={42,8,-6, 10,10,10};
    hiopMatrixDense res(m,k); res.copyFrom(true_result);

    C.addMatrix(-1.0,res);
    double Cmax=C.max_abs_value();
    if(fabs(Cmax)>1e-14) {
      printf("timesMat test failed\n");
      return false;
    }
  }
  // C = beta*C + alpha*A'*B
  {
    m=2; n=5; k=3;
    double c[]={0.0,0.0,0.0, 6,12,24};
    hiopMatrixDense C(m,k); C.copyFrom(c);

    double a[]={1,0.5, 2,0.5, 3,0.5, 4,0.5, 5,0.5};
    hiopMatrixDense A(n,m); A.copyFrom(a);
    
    double b[]={1,4,-1, 2,0,-1, 1,0,0, 2,0,0, 1,0,0};
    hiopMatrixDense B(n,k); B.copyFrom(b);

    A.transTimesMat(0.5, C, 2.0, B);
    C.print();
    
    double true_result[]={42,8,-6, 10,10,10};
    hiopMatrixDense res(m,k); res.copyFrom(true_result);

    C.addMatrix(-1.0,res);
    double Cmax=C.max_abs_value();
    if(fabs(Cmax)>1e-14) {
      printf("transTimesMat test failed\n");
      return false;
    }
  }
  // C= beta*C + alpha*A*B'
  {
    m=2; n=5; k=3;
    double c[]={0.0,0.0,0.0, 6,12,24};
    hiopMatrixDense C(m,k); C.copyFrom(c);

    double a[]={1,2,3,4,5, 0.5,0.5,0.5,0.5,0.5};
    hiopMatrixDense A(m,n); A.copyFrom(a);
    
    double b[]={1,2,1,2,1, 4,0,0,0,0, -1,-1,0,0,0};
    hiopMatrixDense B(k,n); B.copyFrom(b);

    A.timesMatTrans(0.5, C, 2.0, B);
    C.print();
    
    double true_result[]={42,8,-6, 10,10,10};
    hiopMatrixDense res(m,k); res.copyFrom(true_result);

    C.addMatrix(-1.0,res);
    double Cmax=C.max_abs_value();
    if(fabs(Cmax)>1e-14) {
      printf("transTimesMat test failed\n");
      return false;
    }
  } 
  return true;
}

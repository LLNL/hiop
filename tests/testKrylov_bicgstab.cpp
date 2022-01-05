#include "hiopKrylovSolver.hpp"
#include "hiopLinAlgFactory.hpp"
#include "hiopVector.hpp"
#include "hiopLinearOperator.hpp"

#include "hiopMatrixSparseTriplet.hpp"

#include <cstdlib>
#include <string>

using namespace hiop;

void initializeSymSparseMat(hiop::hiopMatrixSparse* mat, bool is_diag_pred)
{
  auto* A = dynamic_cast<hiop::hiopMatrixSymSparseTriplet*>(mat);
  size_type* iRow = A->i_row();
  size_type* jCol = A->j_col();
  double* val = A->M();
  const auto nnz = A->numberOfNonzeros();
  int nonZerosUsed = 0;

  size_type m = A->m();
  size_type n = A->n();

  int num_entries = n * m - (m * (m - 1) / 2);
  int density = num_entries / nnz;

  auto iRow_idx = 0;
  auto jCol_idx = 0;

  for (auto i = 0; i < m; i++)
  {
    iRow[nonZerosUsed] = i;
    jCol[nonZerosUsed] = i;
    if(is_diag_pred) {
      val[nonZerosUsed] = 1.0/((i+1.0)*5.);
    } else {
      val[nonZerosUsed] = (i+1.0)*5.;
    }
    nonZerosUsed++;
#if 1
    if(!is_diag_pred) {
      if(i+1<m) {
        iRow[nonZerosUsed] = i;
        jCol[nonZerosUsed] = i+1;
        val[nonZerosUsed] = (i+1.0)*2.;
        nonZerosUsed++;        
      }

      if(i+2<m) {
        iRow[nonZerosUsed] = i;
        jCol[nonZerosUsed] = i+2;
        val[nonZerosUsed] = (i+1.0)*1.;
        nonZerosUsed++;        
      }     
    }
#endif
  }
  assert(nnz == nonZerosUsed && "incorrect amount of non-zeros in sparse sym matrix");
}

int main(int argc, char **argv)
{
  size_type n = 50;
  
  if(argc>1) {
    n = std::atoi(argv[1]);
    if(n<=0) {
      n = 50;
    }
  }
  
  printf("\nTesting hiopBiCGStabSolver with matrix_%dx%d\n",n,n);

  const std::string mem_space = "DEFAULT";

  size_type M_local = n;
  size_type N_local = M_local;
  size_type nnz = M_local + M_local-1 + M_local-2;

  hiop::hiopVectorPar rhs(N_local);
  rhs.setToConstant(1.0);

  // create a sysmetric matrix (only upper triangular part is needed by hiop)
  // it is an upper tridiagonal matrix
  hiop::hiopMatrixSparse* A_mat = 
    hiop::LinearAlgebraFactory::create_matrix_sym_sparse(mem_space, M_local, nnz);
  initializeSymSparseMat(A_mat, false);

  // use the diagonal part as a preconditioner
  // build the inverse of the diagonal preconditioner as a simple hiopLinearOperator
  hiop::hiopMatrixSparse* Minv_mat = 
    hiop::LinearAlgebraFactory::create_matrix_sym_sparse(mem_space, M_local, N_local);
  initializeSymSparseMat(Minv_mat, true);

  hiopMatVecOpr* A_opr = new hiopMatVecOpr(A_mat);
  hiopMatVecOpr* Minv_opr = new hiopMatVecOpr(Minv_mat);
  
  hiopBiCGStabSolver solver(nullptr,A_opr,Minv_opr,nullptr,nullptr);
  
  bool is_solved = solver.solve(rhs);

  // Destroy testing objects
  delete A_opr;
  delete Minv_opr;
  delete A_mat;
  delete Minv_mat;
  return 0;
}


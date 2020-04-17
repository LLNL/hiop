#include "hiopKronReduction.hpp"

#include "hiopLinSolverUMFPACKZ.hpp"
#include "hiopCppStdUtils.hpp"

namespace hiop
{

  bool hiopKronReduction::go(const std::vector<int>& idx_nonaux_buses, 
			     const std::vector<int>& idx_aux_buses,
			     const hiopMatrixComplexSparseTriplet& Ybus, 
			     hiopMatrixComplexDense& Ybus_red)
  {
    //printvec(idx_aux_buses, "aux=");
    //printvec(idx_nonaux_buses, "nonaux=");

    //Ybus.print(); 
    //int nnz = Ybus.numberOfNonzeros();
    //printf("Ybus has %d nnz\n", nnz);
    
    //Yaa = Matrix(Ybus[nonaux, nonaux])
    auto* Yaa = Ybus.new_slice(idx_nonaux_buses.data(),
			       idx_nonaux_buses.size(),
			       idx_nonaux_buses.data(),
			       idx_nonaux_buses.size());

    auto* Ybb = Ybus.new_slice(idx_aux_buses.data(),
			       idx_aux_buses.size(),
			       idx_aux_buses.data(),
			       idx_aux_buses.size());
    
    auto* Yba = Ybus.new_slice(idx_aux_buses.data(),
			       idx_aux_buses.size(),
			       idx_nonaux_buses.data(),
			       idx_nonaux_buses.size());
    //Yba->print();
    //Ybb->print();
    //fflush(stdout);
    hiopLinSolverUMFPACKZ* linsolver = new hiopLinSolverUMFPACKZ(*Ybb);

    int nret = linsolver->matrixChanged();
    if(nret>=0) {

      //
      //Yaa - Yab*(Ybb\Yba)
      //

      //Ybb\Yba
      hiopMatrixComplexDense Ybbinv_Yba(Yba->m(), Yba->n());
      linsolver->solve(*Yba, Ybbinv_Yba);
      //Ybbinv_Yba.print();
      
      delete Ybb;
      delete linsolver;

      //Ybus_red = - Yab*(Ybb\Yba)
      Yba->transTimesMat(0.0, Ybus_red, -1.0, Ybbinv_Yba);
      delete Yba;

      //Ybus_red.addSparseSymUpperTriangleToSymDenseMatrixUpperTriangle(1.0, *Yaa);
      Ybus_red.addSparseMatrix(std::complex<double>(1.0, 0.0), *Yaa);
      delete Yaa;
      //Ybus_red.print();

    } else {
      printf("Error occured while performing the Kron reduction (factorization issue)\n");
      delete linsolver;
      delete Yaa;
      delete Ybb;
      delete Yba;
      return false;
    }
    return true;
  }
}//end namespace

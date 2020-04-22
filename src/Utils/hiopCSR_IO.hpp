#ifndef HIOP_CSR_IO
#define HIOP_CSR_IO

#include <string>
#ifdef HIOP_USE_MPI
#include <mpi.h>
#endif

namespace hiop
{
  //saves a dense or other matrices in the CSR format. Expects the following order of calls
  // 1. writeMatToFile -> will create/overwrite kkt_linsys_counter.iajaaa file and will write 
  // the matrix  passed as argument
  // 2. writeRhsToFile -> will append the rhs
  // 3. writeSolToFile -> will append the sol
  class hiopCSR_IO {
  public:
    // masterrank=-1 means all ranks save
    hiopCSR_IO(hiopNlpFormulation* nlp, int masterrank=0)
      : _nlp(nlp), _master_rank(masterrank), _f(NULL), m(-1), last_counter(-1)
    {
    }

    virtual ~hiopCSR_IO() 
    {
    }

    void writeRhsToFile(const hiopVectorPar& rhs, const int& counter)
    {
#ifdef HIOP_USE_MPI
      if(_master_rank>=0 && _master_rank != _nlp->get_rank()) return;
#endif
      assert(counter == last_counter);
      assert(m == rhs.get_size());

      std::string fname = "kkt_linsys_"; 
      fname += std::to_string(counter); 
      fname += ".iajaaa";
      FILE* f = fopen(fname.c_str(), "a+");
      if(NULL==f) {
	_nlp->log->printf(hovError, "Could not open '%s' for writing the rhs/sol.\n", fname.c_str());
	return;
      }

      const double* v = rhs.local_data_const();
      for(int i=0; i<m; i++)
	fprintf(f, "%.20f ", v[i]);
      fprintf(f, "\n");
      fclose(f);
    }
    inline void writeSolToFile(const hiopVectorPar& sol, const int& counter)
    { 
      writeRhsToFile(sol, counter); 
    }

    //write a dense matrix in the iajaaa format; zero elements are not written
    //counter specifies the suffix in the filename, essentially is the iteration #
    void writeMatToFile(hiopMatrixDense& Msys, const int& counter)
    {
#ifdef HIOP_USE_MPI
      if(_master_rank>=0 && _master_rank != _nlp->get_rank()) return;
#endif
      last_counter = counter;
      m = Msys.m();

      std::string fname = "kkt_linsys_"; 
      fname += std::to_string(counter); 
      fname += ".iajaaa";
      FILE* f = fopen(fname.c_str(), "w+");
      if(NULL==f) {
	_nlp->log->printf(hovError, "Could not open '%s' for writing the linsys.\n", fname.c_str());
	return;
      }
      
      //count nnz
      const double zero_tol = 1e-25;
      int nnz=0;
      double** M = Msys.local_data();
      for(int i=0; i<m; i++) for(int j=i; j<m; j++) if(fabs(M[i][j])>zero_tol) nnz++;
      
      //start writing -> indexes are starting at 1
      fprintf(f, "%d\n %d\n", m, nnz);
      
      //array of pointers/offsets in of the first nonzero of each row; first entry is 1 and the last entry is nnz+1
      int offset = 1;
      fprintf(f, "%d ", offset);
      for(int i=0; i<m; i++) {
	for(int j=i; j<m; j++) 
	  if(fabs(M[i][j])>zero_tol)
	    offset++;
	
	fprintf(f, "%d ", offset);
      }
      assert(offset == nnz+1);
      fprintf(f, "\n");
      
      //array of the column indexes of nonzeros
      for(int i=0; i<m; i++) {
	for(int j=i; j<m; j++) 
	  if(fabs(M[i][j])>zero_tol)
	    fprintf(f, "%d ", j);
    }
      fprintf(f, "\n");
      
      //array of nonzero entries of the matrix
      for(int i=0; i<m; i++) {
	for(int j=i; j<m; j++) 
	  if(fabs(M[i][j])>zero_tol)
	    fprintf(f, "%.20f ", M[i][j]);
      }
      fprintf(f, "\n");
      
      fclose(f);
    }
  private:
    FILE* _f;
    hiopNlpFormulation* _nlp;
    int _master_rank;
    int m, last_counter; //used only for consistency (such as order of calls) checks
  };
} // end namespace


#endif

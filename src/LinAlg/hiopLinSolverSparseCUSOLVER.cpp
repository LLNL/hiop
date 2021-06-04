#include "hiopLinSolverSparseCUSOLVER.hpp"

#include "hiop_blasdefs.hpp"
//#include "klu_parse.hxx"

#include "cholmod.h"
#include "klu.h"
#include "cusparse_v2.h"
namespace hiop
{

  hiopLinSolverIndefSparseCUSOLVER::hiopLinSolverIndefSparseCUSOLVER(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverIndefSparse(n, nnz, nlp),
    kRowPtr_{nullptr},jCol_{nullptr},kVal_{nullptr},index_covert_CSR2Triplet_{nullptr},index_covert_extra_Diag2CSR_{nullptr},
    n_{n}, nnz_{0}
  {
  }

  hiopLinSolverIndefSparseCUSOLVER::~hiopLinSolverIndefSparseCUSOLVER()
  {
    if(kRowPtr_)
      delete [] kRowPtr_;
    if(jCol_)
      delete [] jCol_;
    if(kVal_)
      delete [] kVal_;
    if(index_covert_CSR2Triplet_)
      delete [] index_covert_CSR2Triplet_;
    if(index_covert_extra_Diag2CSR_)
      delete [] index_covert_extra_Diag2CSR_;
    //KS: make sure we delete the GPU variables
#if 0
    cudaFree(dia);
    cudaFree(da);
    cudaFree(dja);
    cudaFree(devr);
    cudaFree(devx);
    cudaFree(d_work);
    cusparseDestroy(handle); 
    cusolverSpDestroy(handle_cusolver);
    cublasDestroy(handle_cublas);
    cusparseDestroyMatDescr(descrA);

    cusparseDestroyMatDescr(descrM);
    cusolverSpDestroyCsrluInfoHost(info_lu);
    cusolverSpDestroyGluInfo(info_M);

    klu_free_symbolic(&Symbolic, &Common) ;
    klu_free_numeric(&Numeric, &Common) ;
#endif 
  }

  void hiopLinSolverIndefSparseCUSOLVER::firstCall()
  {

    assert(n_==M.n() && M.n()==M.m());
    assert(n_>0);

    kRowPtr_ = new int[n_+1]{0};

    // transfer triplet form to CSR form
    // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the additional diagonal elememts
    // the 1st part is sorted by row
    {
      //
      // compute nnz in each row
      //
      // off-diagonal part
      kRowPtr_[0]=0;
      for(int k=0;k<M.numberOfNonzeros()-n_;k++){
        if(M.i_row()[k]!=M.j_col()[k]){
          kRowPtr_[M.i_row()[k]+1]++;
          kRowPtr_[M.j_col()[k]+1]++;
          nnz_ += 2;
        }
      }
      // diagonal part
      for(int i=0;i<n_;i++){
        kRowPtr_[i+1]++;
        nnz_ += 1;
      }
      // get correct row ptr index
      for(int i=1;i<n_+1;i++){
        kRowPtr_[i] += kRowPtr_[i-1];
      }
      assert(nnz_==kRowPtr_[n_]);

      kVal_ = new double[nnz_]{0.0};
      jCol_ = new int[nnz_]{0};

    }
    {
      //
      // set correct col index and value
      //
      index_covert_CSR2Triplet_ = new int[nnz_];
      index_covert_extra_Diag2CSR_ = new int[n_];

      int *nnz_each_row_tmp = new int[n_]{0};
      int total_nnz_tmp{0},nnz_tmp{0}, rowID_tmp, colID_tmp;
      for(int k=0;k<n_;k++) index_covert_extra_Diag2CSR_[k]=-1;

      for(int k=0;k<M.numberOfNonzeros()-n_;k++){
        rowID_tmp = M.i_row()[k];
        colID_tmp = M.j_col()[k];
        if(rowID_tmp==colID_tmp){
          nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
          jCol_[nnz_tmp] = colID_tmp;
          kVal_[nnz_tmp] = M.M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          kVal_[nnz_tmp] += M.M()[M.numberOfNonzeros()-n_+rowID_tmp];
          index_covert_extra_Diag2CSR_[rowID_tmp] = nnz_tmp;

          nnz_each_row_tmp[rowID_tmp]++;
          total_nnz_tmp++;
        }else{
          nnz_tmp = nnz_each_row_tmp[rowID_tmp] + kRowPtr_[rowID_tmp];
          jCol_[nnz_tmp] = colID_tmp;
          kVal_[nnz_tmp] = M.M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          nnz_tmp = nnz_each_row_tmp[colID_tmp] + kRowPtr_[colID_tmp];
          jCol_[nnz_tmp] = rowID_tmp;
          kVal_[nnz_tmp] = M.M()[k];
          index_covert_CSR2Triplet_[nnz_tmp] = k;

          nnz_each_row_tmp[rowID_tmp]++;
          nnz_each_row_tmp[colID_tmp]++;
          total_nnz_tmp += 2;
        }
      }
      // correct the missing diagonal term
      for(int i=0;i<n_;i++){
        if(nnz_each_row_tmp[i] != kRowPtr_[i+1]-kRowPtr_[i]){
          assert(nnz_each_row_tmp[i] == kRowPtr_[i+1]-kRowPtr_[i]-1);
          nnz_tmp = nnz_each_row_tmp[i] + kRowPtr_[i];
          jCol_[nnz_tmp] = i;
          kVal_[nnz_tmp] = M.M()[M.numberOfNonzeros()-n_+i];
          index_covert_CSR2Triplet_[nnz_tmp] = M.numberOfNonzeros()-n_+i;
          total_nnz_tmp += 1;

          std::vector<int> ind_temp(kRowPtr_[i+1]-kRowPtr_[i]);
          std::iota(ind_temp.begin(), ind_temp.end(), 0);
          std::sort(ind_temp.begin(), ind_temp.end(),[&](int a, int b){ return jCol_[a+kRowPtr_[i]]<jCol_[b+kRowPtr_[i]]; });

          reorder(kVal_+kRowPtr_[i],ind_temp,kRowPtr_[i+1]-kRowPtr_[i]);
          reorder(index_covert_CSR2Triplet_+kRowPtr_[i],ind_temp,kRowPtr_[i+1]-kRowPtr_[i]);
          std::sort(jCol_+kRowPtr_[i],jCol_+kRowPtr_[i+1]);
        }
      }

      delete   [] nnz_each_row_tmp;
    }


    /*
     * initialize KLU and cuSolver parameters
     */
    //handles

    cusparseCreate(&handle); 
    cusolverSpCreate(&handle_cusolver);
    cublasCreate(&handle_cublas);

    //descriptors
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&descrM);
    cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO);

    //info (data structure where factorization is stored)
    cusolverSpDestroyCsrluInfoHost(info_lu);
    cusolverSpDestroyGluInfo(info_M);
    cusolverSpCreateCsrluInfoHost(&info_lu);
    cusolverSpCreateGluInfo(&info_M);

    //KLU 

    klu_defaults(&Common) ;

    //modify as you please
    //KS: consider making a part of setup options that can be called from a user side

    Common.btf = 0;
    Common.ordering = 1;//COLAMD; use 0 for AMD
    Common.tol = 0.01;
    Common.scale = 0;

    //allocate gpu data
    free(mia); free(mja);
    mia=NULL; mja = NULL;
    cudaFree(devx);
    cudaFree(devr);
    devx = NULL;
    devr=NULL;
    checkCudaErrors(cudaMalloc(&devx, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr, n_ * sizeof(double)));
    this->newKLUfactorization();
  }

//helper private function needed for format conversion

int hiopLinSolverIndefSparseCUSOLVER::createM(const int n, 
          const int nnzL, 
          const int *Lp, 
          const int *Li,
          const int nnzU, 
          const int *Up, 
          const int *Ui){

  int row;
  for (int i = 0; i<n; ++i){
    //go through EACH COLUMN OF L first
    for (int j = Lp[i]; j<Lp[i+1]; ++j){
      row = Li[j];
      //BUT dont count diagonal twice, important
      if (row != i){
	mia[row+1]++;
      }
    }
    //then each column of U
    for (int j = Up[i]; j<Up[i+1]; ++j){
      row = Ui[j];
      mia[row+1]++;
    }
  }
  //then organize mia;
  mia[0] = 0;
  for (int i=1; i<n+1; i++){
    mia[i]+=mia[i-1];
  } 

  int *Mshifts = (int *) calloc (n,sizeof(int));
  for (int i = 0; i<n; ++i){

    //go through EACH COLUMN OF L first
    for (int j = Lp[i]; j<Lp[i+1]; ++j){
      row = Li[j];
      if (row != i){
	//place (row, i) where it belongs!

	mja[mia[row]+Mshifts[row]] = i;
	Mshifts[row]++;
      }
    }
//each column of U next
    for (int j = Up[i]; j<Up[i+1]; ++j){
      row = Ui[j];
      mja[mia[row]+Mshifts[row]] = i;
      Mshifts[row]++;
    }

  }
  free(Mshifts);


}


  // call if both the matrix and the nnz structure changed or if convergence is poor while using refactorization.
  int hiopLinSolverIndefSparseCUSOLVER::newKLUfactorization()
  {
    klu_free_symbolic(&Symbolic, &Common) ;
    klu_free_numeric(&Numeric, &Common) ;
    Symbolic = klu_analyze(n_, kRowPtr_, jCol_, &Common) ;
    if (Symbolic == NULL){
      //  klu_error_codes(Common);
      //throw error    

      printf("symbolic NULL\n");
    }

    Numeric = klu_factor(kRowPtr_, jCol_, kVal_, Symbolic, &Common);
    if (Numeric == NULL){
      printf("numeric NULL \n");
      //throw error    
    }


    // get sizes


    const int nnzL = Numeric->lnz;
    const int nnzU = Numeric->unz;

    const int nnzM = (nnzL+nnzU-n_);
    /* parse the factorization */

    if (mia !=NULL) {
      free(mia);
}
    if (mja !=NULL) {
      free(mja);
}

    mia = (int*)calloc(sizeof(int),(n_+1));
    mja = (int*)calloc(sizeof(int),nnzM);

    if (P==NULL){
      P = (int *)malloc(sizeof(int) * n_);
      Q = (int *)malloc(sizeof(int) * n_);
    }
//KS new code

       memcpy(P, Numeric->Pnum, sizeof(int)*n_);
       memcpy(Q, Symbolic->Q, sizeof(int)*n_);
      int * Lp = (int*)malloc((n_+1)*sizeof(int));
      int * Li = (int*)malloc(nnzL*sizeof(int));
//we cant use NULL instrad od Lx and Ux because it causes SEG FAULT. It seems like a waste of memory though.      
      double * Lx = (double*)malloc(nnzL*sizeof(double));
      int * Up = (int*)malloc((n_+1)*sizeof(int));
      int * Ui = (int*)malloc(nnzU * sizeof(int));
      double *Ux = (double*)malloc(nnzU * sizeof(double));

      int ok = klu_extract(Numeric, Symbolic, Lp, Li, Lx, Up, Ui, Ux, 
	  NULL, NULL, NULL, 
	  NULL, NULL, 
	  NULL, NULL, &Common);

      createM(n_, nnzL, Lp, Li,nnzU, Up, Ui);
free(Lp); free(Li); 
free(Lx);
free(Up); free(Ui); 
free(Ux);

//KS: to be changed
#if 0    
  int st = parse_klu(
          Symbolic,
          Numeric,
          &Common,
          n_,
          /* A is CSR, base-0 */
          nnz_,
          kRowPtr_,
          jCol_,
          kVal_,
          /* M is CSR, base-0 */
          nnzM,
          mia,  /* <int> n+1 */
          mja,  /* <int> nnzM */
          P,  /* <int> n */
          Q   /* <int> n */
          );
#endif

    /* setup GLU */ 
    sp_status = cusolverSpDgluSetup(
        handle_cusolver,
        n_,
        nnz_,
        descrA,
        kRowPtr_,
        jCol_,
        P, /* base-0 */
        Q, /* base-0 */
        nnzM, /* nnzM */
        descrM,
        mia,
        mja,
        info_M);
    sp_status = cusolverSpDgluBufferSize(
        handle_cusolver,
        info_M,
        &size_M);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status);
    bufferSize = size_M;
    if (d_work !=NULL)cudaFree(d_work);
    //doesnt work
    cudaMalloc((void **)&d_work, bufferSize);
    sp_status = cusolverSpDgluAnalysis(
        handle_cusolver,
        info_M,
        d_work);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status);

    //now make sure the space is allocated for A on the GPU (but dont copy)
    if (da != NULL) cudaFree(da); 
    if (dja != NULL) cudaFree(da);
    if (dia != NULL) cudaFree(dia);
    //dont free d_ia -> size is n+1 so doesnt matter
    checkCudaErrors(cudaMalloc(&da, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dja, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dia, (n_ +1)* sizeof(int)));
    //copy    

    checkCudaErrors(cudaMemcpy(da, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dia, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dja, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));
    //reset and refactor so factors are ON THE GPU

    sp_status = cusolverSpDgluReset(
        handle_cusolver,
        n_,
        /* A is original matrix */
        nnz_,
        descrA,
        da,
        dia,
        dja,
        info_M);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status);
    sp_status = cusolverSpDgluFactor(
        handle_cusolver,
        info_M,
        d_work);
  }

  int hiopLinSolverIndefSparseCUSOLVER::matrixChanged()
  {
    assert(n_==M.n() && M.n()==M.m());
    assert(n_>0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if( !kRowPtr_ ){
      this->firstCall();
    }else{
      // update matrix
      int rowID_tmp{0};
      for(int k=0;k<nnz_;k++){
        kVal_[k] = M.M()[index_covert_CSR2Triplet_[k]];
      }
      for(int i=0;i<n_;i++){
        if(index_covert_extra_Diag2CSR_[i] != -1)
          kVal_[index_covert_extra_Diag2CSR_[i]] += M.M()[M.numberOfNonzeros()-n_+i];
      }
      // somehow update the matrix not sure how
      //     spss.set_csr_matrix(n_, kRowPtr_, jCol_, kVal_, true);

      //call newfactorization if necessart
      //update the GPU matrix

      checkCudaErrors(cudaMemcpy(da, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
      //  checkCudaErrors(cudaMemcpy(dia, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
      //  checkCudaErrors(cudaMemcpy(dja, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));

      //re-factor here

      sp_status = cusolverSpDgluReset(
          handle_cusolver,
          n_,
          /* A is original matrix */
          nnz_,
          descrA,
          da,
          dia,
          dja,
          info_M);


      sp_status = cusolverSpDgluFactor(
          handle_cusolver,
          info_M,
          d_work);

      //end of factor
    }    
    nlp_->runStats.linsolv.tmInertiaComp.start();
    int negEigVal = nFakeNegEigs_;
    nlp_->runStats.linsolv.tmInertiaComp.stop();
    return negEigVal;
  }

  bool hiopLinSolverIndefSparseCUSOLVER::solve ( hiopVector& x )
  {
    assert(n_==M.n() && M.n()==M.m());
    assert(n_>0);
    assert(x.get_size()==M.n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

  //  hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    
    hiopVector* rhs = x.new_copy();
    double* dx = x.local_data();
    double* drhs = rhs->local_data();
    //copy x, rhs to the GPU

#if 1
    checkCudaErrors(cudaMemcpy(devr, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

    //  spss.solve(drhs, dx);
    //solve HERE

    sp_status = cusolverSpDgluSolve(
        handle_cusolver,
        n_,
        /* A is original matrix */
        nnz_,
        descrA,
        da,
        dia,
        dja,
        devr, /* right hand side */
        devx,   /* left hand side */
        &ite_refine_succ,
        &r_nrminf,
        info_M,
        d_work);
    //copy the solutuion back
    if (sp_status == 0){
      checkCudaErrors(cudaMemcpy(dx, devx, sizeof(double) * n_, cudaMemcpyDeviceToHost));
    }
    else {
      printf("solve error! \n");
    }
    nlp_->runStats.linsolv.tmTriuSolves.stop();

    delete rhs; rhs=nullptr;
    return 1;
#endif
  }

  //the NonSym version.

  hiopLinSolverNonSymSparseCUSOLVER::hiopLinSolverNonSymSparseCUSOLVER(const int& n, const int& nnz, hiopNlpFormulation* nlp)
    : hiopLinSolverNonSymSparse(n, nnz, nlp),
    kRowPtr_{nullptr},jCol_{nullptr},kVal_{nullptr},index_covert_CSR2Triplet_{nullptr},index_covert_extra_Diag2CSR_{nullptr},
    n_{n}, nnz_{0}
  {}

  hiopLinSolverNonSymSparseCUSOLVER::~hiopLinSolverNonSymSparseCUSOLVER()
  {
    if(kRowPtr_)
      delete [] kRowPtr_;
    if(jCol_)
      delete [] jCol_;
    if(kVal_)
      delete [] kVal_;
    if(index_covert_CSR2Triplet_)
      delete [] index_covert_CSR2Triplet_;
    if(index_covert_extra_Diag2CSR_)
      delete [] index_covert_extra_Diag2CSR_;
    //KS: make sure we delete the GPU variables
#if 0  
    cudaFree(dia);
    cudaFree(da);
    cudaFree(dja);
    cudaFree(devr);
    cudaFree(devx);
#endif 
  }

int hiopLinSolverNonSymSparseCUSOLVER::createM(const int n, 
          const int nnzL, 
          const int *Lp, 
          const int *Li,
          const int nnzU, 
          const int *Up, 
          const int *Ui){

  int row;
  for (int i = 0; i<n; ++i){
    //go through EACH COLUMN OF L first
    for (int j = Lp[i]; j<Lp[i+1]; ++j){
      row = Li[j];
      //BUT dont count diagonal twice, important
      if (row != i){
	mia[row+1]++;
      }
    }
    //then each column of U
    for (int j = Up[i]; j<Up[i+1]; ++j){
      row = Ui[j];
      mia[row+1]++;
    }
  }
  //then organize mia;
  mia[0] = 0;
  for (int i=1; i<n+1; i++){
    mia[i]+=mia[i-1];
  } 

  int *Mshifts = (int *) calloc (n,sizeof(int));
  for (int i = 0; i<n; ++i){

    //go through EACH COLUMN OF L first
    for (int j = Lp[i]; j<Lp[i+1]; ++j){
      row = Li[j];
      if (row != i){
	//place (row, i) where it belongs!

	mja[mia[row]+Mshifts[row]] = i;
	Mshifts[row]++;
      }
    }
//each column of U next
    for (int j = Up[i]; j<Up[i+1]; ++j){
      row = Ui[j];
      mja[mia[row]+Mshifts[row]] = i;
      Mshifts[row]++;
    }

  }
  free(Mshifts);


}
  void hiopLinSolverNonSymSparseCUSOLVER::firstCall()
  {
    assert(n_==M.n() && M.n()==M.m());
    assert(n_>0);

    // transfer triplet form to CSR form
    // note that input is in lower triangular triplet form. First part is the sparse matrix, and the 2nd part are the additional diagonal elememts
    // the 1st part is sorted by row

    M.convertToCSR(nnz_, &kRowPtr_, &jCol_, &kVal_, &index_covert_CSR2Triplet_, &index_covert_extra_Diag2CSR_, extra_diag_nnz_map);

    /*
     * initialize cusolver parameters
     */

    //handles

    cusparseCreate(&handle); 
    cusolverSpCreate(&handle_cusolver);
    cublasCreate(&handle_cublas);

    //descriptors
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&descrM);
    cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrM, CUSPARSE_INDEX_BASE_ZERO);

    //info (data structure where factorization is stored)
    cusolverSpCreateCsrluInfoHost(&info_lu);
    cusolverSpCreateGluInfo(&info_M);

    //KLU 

    klu_defaults(&Common) ;

    //modify as you please
    //KS: consider making a part of setup options that can be called from a user side

    Common.btf = 0;
    Common.ordering = 1;//COLAMD; use 0 for AMD
    Common.tol = 0.01;
    Common.scale = 0;
    free(mia); free(mja);
    mia = NULL;
    mja = NULL;
    //allocate gpu data

    cudaFree(devx);
    cudaFree(devr);
    devx = NULL;
    devr=NULL;
    checkCudaErrors(cudaMalloc(&devx, n_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&devr, n_ * sizeof(double)));
    this->newKLUfactorization();
  }


  int hiopLinSolverNonSymSparseCUSOLVER::newKLUfactorization()
  {

    Symbolic = klu_analyze(n_, kRowPtr_, jCol_, &Common) ;

    if (Symbolic == NULL){
      //  klu_error_codes(Common);
      //throw error    
      printf("symbolic NULL"); 
    }

    Numeric = klu_factor(kRowPtr_, jCol_, kVal_, Symbolic, &Common);
    if (Numeric == NULL){
      //klu_error_codes(Common);
      //throw error    
      printf("numeric  NULL"); 
    }


    // get sizes


    const int nnzL = Numeric->lnz;
    const int nnzU = Numeric->unz;

    const int nnzM = (nnzL+nnzU-n_);
    /* parse the factorization */

    if (mia !=NULL) free(mia);
    if (mja !=NULL) free(mja);

    mia = (int*)calloc(sizeof(int),(n_+1));
    mja = (int*)calloc(sizeof(int),nnzM);

    if (P==NULL){
      P = (int *)malloc(sizeof(int) * n_);
      Q = (int *)malloc(sizeof(int) * n_);
    }
//createM GOES HERE
//
       memcpy(P, Numeric->Pnum, sizeof(int)*n_);
       memcpy(Q, Symbolic->Q, sizeof(int)*n_);
      int * Lp = (int*)malloc((n_+1)*sizeof(int));
      int * Li = (int*)malloc(nnzL*sizeof(int));
//we cant use NULL instrad od Lx and Ux because it causes SEG FAULT. It seems like a waste of memory though.      
      double * Lx = (double*)malloc(nnzL*sizeof(double));
      int * Up = (int*)malloc((n_+1)*sizeof(int));
      int * Ui = (int*)malloc(nnzU * sizeof(int));
      double *Ux = (double*)malloc(nnzU * sizeof(double));

      int ok = klu_extract(Numeric, Symbolic, Lp, Li, Lx, Up, Ui, Ux, 
	  NULL, NULL, NULL, 
	  NULL, NULL, 
	  NULL, NULL, &Common);

      createM(n_, nnzL, Lp, Li,nnzU, Up, Ui);
free(Lp); free(Li); 
free(Lx);
free(Up); free(Ui); 
free(Ux);

    /* setup GLU */ 
    sp_status = cusolverSpDgluSetup(
        handle_cusolver,
        n_,
        nnz_,
        descrA,
        kRowPtr_,
        jCol_,
        P, /* base-0 */
        Q, /* base-0 */
        nnzM, /* nnzM */
        descrM,
        mia,
        mja,
        info_M);
    sp_status = cusolverSpDgluBufferSize(
        handle_cusolver,
        info_M,
        &size_M);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status);
    bufferSize = size_M;
    if (d_work !=NULL)cudaFree(d_work);
    //doesnt work
    cudaMalloc((void **)&d_work, bufferSize);
    sp_status = cusolverSpDgluAnalysis(
        handle_cusolver,
        info_M,
        d_work);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status);

    //now make sure the space is allocated for A on the GPU (but dont copy)
    if (da != NULL) cudaFree(da); 
    if (dja != NULL) cudaFree(da);
    if (dia != NULL) cudaFree(dia);
    checkCudaErrors(cudaMalloc(&da, nnz_ * sizeof(double)));
    checkCudaErrors(cudaMalloc(&dja, nnz_ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&dia, (n_ +1)* sizeof(int)));
    //dont free d_ia -> size is n+1 so doesnt matter
    if (dia==NULL) {
      checkCudaErrors(cudaMalloc(&dia, (n_ +1)* sizeof(int)));
    }
    checkCudaErrors(cudaMemcpy(da, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dia, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dja, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));


    //reset and refactor so factors are ON THE GPU

    sp_status = cusolverSpDgluReset(
        handle_cusolver,
        n_,
        /* A is original matrix */
        nnz_,
        descrA,
        da,
        dia,
        dja,
        info_M);
    assert(CUSOLVER_STATUS_SUCCESS == sp_status);
    sp_status = cusolverSpDgluFactor(
        handle_cusolver,
        info_M,
        d_work);

  }

  int hiopLinSolverNonSymSparseCUSOLVER::matrixChanged()
  {
    assert(n_==M.n() && M.n()==M.m());
    assert(n_>0);

    nlp_->runStats.linsolv.tmFactTime.start();

    if( !kRowPtr_ ){
      this->firstCall();
    }else{
      // update matrix
      int rowID_tmp{0};
      for(int k=0;k<nnz_;k++){
        kVal_[k] = M.M()[index_covert_CSR2Triplet_[k]];
      }
      for(int i=0;i<n_;i++){
        if(index_covert_extra_Diag2CSR_[i] != -1)
          kVal_[index_covert_extra_Diag2CSR_[i]] += M.M()[M.numberOfNonzeros()-n_+i];
      }

      checkCudaErrors(cudaMemcpy(da, kVal_, sizeof(double) * nnz_, cudaMemcpyHostToDevice));
      //  checkCudaErrors(cudaMemcpy(dia, kRowPtr_, sizeof(int) * (n_ + 1), cudaMemcpyHostToDevice));
      //checkCudaErrors(cudaMemcpy(dja, jCol_, sizeof(int) * nnz_, cudaMemcpyHostToDevice));

      //re-factor here

      sp_status = cusolverSpDgluReset(
          handle_cusolver,
          n_,
          /* A is original matrix */
          nnz_,
          descrA,
          da,
          dia,
          dja,
          info_M);



      sp_status = cusolverSpDgluReset(
          handle_cusolver,  n_,
          /* A is original matrix */
          nnz_,
          descrA,
          da,
          dia,
          dja,
          info_M);
      //end of factor
      //   spss.set_csr_matrix(n_, kRowPtr_, jCol_, kVal_, true);
    }

    //spss.factor();   // not really necessary, called if needed by solve

    nlp_->runStats.linsolv.tmInertiaComp.start();
    int negEigVal = nFakeNegEigs_;
    nlp_->runStats.linsolv.tmInertiaComp.stop();
    return negEigVal;
  }

  bool hiopLinSolverNonSymSparseCUSOLVER::solve(hiopVector& x_)
  {
    assert(n_==M.n() && M.n()==M.m());
    assert(n_>0);
    assert(x_.get_size()==M.n());

    nlp_->runStats.linsolv.tmTriuSolves.start();

    hiopVectorPar* x = dynamic_cast<hiopVectorPar*>(&x_);
    assert(x != NULL);
    hiopVectorPar* rhs = dynamic_cast<hiopVectorPar*>(x->new_copy());
    double* dx = x->local_data();
    double* drhs = rhs->local_data();

    //    spss.solve(drhs, dx);
#if 1
    checkCudaErrors(cudaMemcpy(devr, drhs, sizeof(double) * n_, cudaMemcpyHostToDevice));

    //  spss.solve(drhs, dx);
    //solve HERE

    sp_status = cusolverSpDgluSolve(
        handle_cusolver,
        n_,
        /* A is original matrix */
        nnz_,
        descrA,
        da,
        dia,
        dja,
        devr, /* right hand side */
        devx,   /* left hand side */
        &ite_refine_succ,
        &r_nrminf,
        info_M,
        d_work);
    //copy the solutuion back

    checkCudaErrors(cudaMemcpy(dx, devx, sizeof(double) * n_, cudaMemcpyDeviceToHost));
    nlp_->runStats.linsolv.tmTriuSolves.stop();
    delete rhs; rhs=nullptr;
    return 1;

#endif
  }


}//namespace hiop

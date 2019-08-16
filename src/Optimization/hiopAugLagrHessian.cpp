#include "hiopAugLagrHessian.hpp"

#include <cassert>
#include <iostream>
#include <cstring>      // memcpy

namespace hiop
{

hiopAugLagrHessian::hiopAugLagrHessian(NLP_CLASS_IN *nlp_in_, int n_vars, int n_slacks, int m_cons, int nnz):
    nlp_in(nlp_in_),
    nvars_nlp(n_vars),
    nslacks_nlp(n_slacks),//TODO: use nlp->getXX() instead of introducing duplicates of variables?
    mcons_nlp(m_cons),
    nnz_nlp(nnz),
    _lambdaForHessEval(nullptr),
    _hessianNlp(nullptr),
    _hessianAugLagr(nullptr),
    _updateIterator(0)
{
    _lambdaForHessEval = new hiopVectorPar(m_cons);
    _hessianNlp = new hiopMatrixSparse(nvars_nlp, nvars_nlp, nnz_nlp);
    _hessianAugLagr = new hiopMatrixSparse(0,0,0);//we don't know nnz at this point
}

hiopAugLagrHessian::~hiopAugLagrHessian()
{
    if(_hessianNlp)              delete _hessianNlp;
    if(_hessianAugLagr)          delete _hessianAugLagr;
    if(_lambdaForHessEval)       delete _lambdaForHessEval;
}

int hiopAugLagrHessian::nnz()
{
    return _hessianAugLagr->nnz();
}

void hiopAugLagrHessian::getStructure(int *iRow, int *jCol)
{
    const int *iRow_src = _hessianAugLagr->get_iRow_const();
    const int *jCol_src = _hessianAugLagr->get_jCol_const();
    const int nnz = _hessianAugLagr->nnz();
    
    memcpy(iRow, iRow_src, nnz*sizeof(int));
    memcpy(jCol, jCol_src, nnz*sizeof(int));
}

void hiopAugLagrHessian::getValues(double*values)
{
    const double *values_src = _hessianAugLagr->get_values_const();
    const int nnz = _hessianAugLagr->nnz();
    
    memcpy(values, values_src, nnz*sizeof(double));
}

/**
 *   Evaluates NLP Hessian and stores it in member #_hessianNlp.
 *   We use lambda =  2*rho*p(x) - lambda in order to account for
 *   contribution not only of the Lagrangian term but also the penalty term.
 *   _hessianNlp = hess_obj + sum_i lambda*H_i,
 *   where H_i are the penalty function Hessians.
 */
bool hiopAugLagrHessian::eval_hess_nlp(const double *x_in, bool new_x, const hiopVectorPar &lambda, const double rho, const hiopVectorPar &penaltyFcn)
{
  double obj_factor = 1.0;

  //initialize the nonzero structure only during the first call
  static bool initializedStructure = false;
  if (!initializedStructure)
  {
    int *iRow_nlp = _hessianNlp->get_iRow();
    int *jCol_nlp = _hessianNlp->get_jCol();

    //hiop::hiopInterfaceDenseConstraints
    bool bret = nlp_in->eval_h(nvars_nlp, nullptr, false,
                obj_factor, mcons_nlp, nullptr, false,
                nnz_nlp, iRow_nlp, jCol_nlp, nullptr);
    assert(bret);
    initializedStructure = true;
    return true;
  }

  // lambdaForHessEval =  2*rho*p(x) - lambda
  _lambdaForHessEval->copyFrom(penaltyFcn);
  _lambdaForHessEval->scale(2*rho);
  _lambdaForHessEval->axpy(-1.0, lambda);

  double *lambdaForHessEval = _lambdaForHessEval->local_data();
  double *values_nlp = _hessianNlp->get_values();

  // Evaluate f(x)_hess + sum_i{ (2*rho*p_i(x) - lambda_i) * p_i(x)_hess}
  //hiop::hiopInterfaceDenseConstraints
  bool bret = nlp_in->eval_h(nvars_nlp, x_in, new_x,
              obj_factor, mcons_nlp, lambdaForHessEval, true,
              nnz_nlp, nullptr, nullptr, values_nlp);
  assert(bret);

  return bret;
}

void hiopAugLagrHessian::assemble(const double *x, bool new_x, double obj_factor,
        const hiopVectorPar &lambda, const double rho, const hiopVectorPar &penaltyFcn,
        const hiopMatrixSparse &penaltyFcn_jacobian, long long *cons_ineq_mapping)
{
    // evaluates NLP hessian #_hessianNlp using
    // lambdaForHessEval =  2*rho*p(x) - lambda
    bool bret = eval_hess_nlp(x, new_x, lambda, rho, penaltyFcn);
    assert(bret);
    
    // test if the structure of the Hessian matrix is initialized.
    // If so we can update directly #.values with the new values
    const bool structureNotInitialized = (_hessianAugLagr->nnz() == 0 &&
                                          _hessianAugLagr->m()==0 &&
                                          _hessianAugLagr->n()==0);

    //tmp storage of the result in case #C is not initialized
    // otherwise we can update directly #C.values
    vector<vector<int>> vvCols(0);
    vector<vector<double>> vvValues(0);
    const int N = nvars_nlp + nslacks_nlp;
    if (structureNotInitialized)
    {
      vvCols.resize(N);
      vvValues.resize(N);
    }

    // _hessianAugLagr = 2*rho*J'J + _hessianNlp
    // _hessianAUgLagr is either an empty matrix (vvCols,vvValues are used)
    // or the nnz values are updated directly in _hessianAugLagr
    transAAplusB(*_hessianAugLagr, vvCols, vvValues, structureNotInitialized,
                 obj_factor*2*rho, penaltyFcn_jacobian,
                 obj_factor,   *_hessianNlp);
  
    //append scaled jacobian and identity (blocks 2-1 and 2-2)
    // H_xx = _hessianAugLagr
    // H_sx = -2*rho*Jineq'
    // H_ss = 2*rho*I
    //     | Hxx   0  |
    // H = |          |
    //     | Hsx  Hss |
    if (nslacks_nlp > 0) {
      appendScaledJacobian(*_hessianAugLagr, vvCols, vvValues, structureNotInitialized,
                         -obj_factor*2*rho, penaltyFcn_jacobian, cons_ineq_mapping);
    }


  //construt the sparse matrix with the result if not done so previously
  if (structureNotInitialized)
    _hessianAugLagr->make(N, N, vvCols, vvValues);

    //FILE *f1=fopen("hessNLP.txt","w");
    //_hessianNlp->print(f1);
    // fclose(f1);

    //FILE *f2=fopen("jac.txt","w");
    //penaltyFcn_jacobian.print(f2);
    // fclose(f2);
    //
    //FILE *f3=fopen("hess.txt","w");
    //_hessianAugLagr->print(f3);
    // fclose(f3);

    //printf("m n nnz %d %d %d\n", _hessianAugLagr->m(), _hessianAugLagr->n(), _hessianAugLagr->nnz());
    //printf("2*rho %g\n", 2*rho);
    //printf("obj_factor %g\n", obj_factor);
    //if (!structureNotInitialized) assert(0);
}


/** C = alpha * A' * A + beta*B
* The method can work with #C being either an empty sparse matrix,
  i.e. hiopMatrixSparse(0,0,0.), in which case the storage is allocated
  and the sparse structure is created. In case #C already contains
  all the required storage space, we only update the numerical values
  of the nonzeros (assuming that the structure was set up previously).
  
  \param[out] C The method computes and returns only the lower triangular part of the symmetric result.
  \param[out] vvCols, vvValues The method computes and returns only the lower triangular part of the symmetric result.
  \param[in] structureNotInitialized Switch deciding which output will be updated, either C or vvCols+vvValues
  \param[in] A is general nonsquare, nonsymmetric matrix
  \param[in] B is square symmetric matrix, containing only lower triangular part
  \param[in] alpha, beta are constants

*/
void hiopAugLagrHessian::transAAplusB(hiopMatrixSparse &C, vector<vector<int>> &vvCols_C, vector<vector<double>> &vvValues_C, bool structureNotInitialized, double alpha, const hiopMatrixSparse &A, double beta, const hiopMatrixSparse &B)
{
  //check input dimensions
  assert(B.m() == B.n());
  assert(A.n() == B.m());

  //data of matrix A
  const int nrows_A      = A.m();
  const int ncols_A      = A.n();
  const int nonzeroes_A  = A.nnz();
  const int *iRow_A      = A.get_iRow_const();
  const int *jCol_A      = A.get_jCol_const();
  const double *values_A = A.get_values_const();

  // data of matrix B
  const int *iRow_B      = B.get_iRow_const();
  const int *jCol_B      = B.get_jCol_const();
  const double *values_B = B.get_values_const();
  const int nonzeroes_B        = B.nnz();
  
  // data of matrix C
  int *iRow_C           = C.get_iRow();
  int *jCol_C           = C.get_jCol();
  double *values_C      = C.get_values();
  const int nonzeroes_C = C.nnz();
  
  //create column respresentation of the matrix A
  //TODO can be reused for all the subsequent calls, except values_A
  vector<vector<int>> vvRows_A(ncols_A); // list of nnz row indices in each column
  vector<vector<double>> vvValues_A(ncols_A); // list of nnz values in each column
  for (int i = 0; i < nonzeroes_A; i++)
  {
    vvRows_A[jCol_A[i]].push_back(iRow_A[i]);
    vvValues_A[jCol_A[i]].push_back(values_A[i]);
  }
  
  _updateIterator = 0; //iterator in C
  int nnz_idx_B = 0; //iterator in B

  // compute alpha*A'A + beta*B
  for (int c1 = 0; c1 < ncols_A; c1++)
  {
    for (int c2 = 0; c2 <= c1; c2++) //compute only lower triangular part
    {
      //TODO: skip empty elements 
      //if (vvRows_A[c1].begin() == vvRows_A.end() && zero @B[c1,c2])

      auto rowIdx1 = vvRows_A[c1].begin();
      auto rowIdx2 = vvRows_A[c2].begin();
      auto value1  = vvValues_A[c1].begin();
      auto value2  = vvValues_A[c2].begin();
      double dot = 0.;
      bool newNonzero = false;

      //compute alpha * A' * A
      while ( rowIdx1 != vvRows_A[c1].end() && rowIdx2 != vvRows_A[c2].end())
      {
        if (*rowIdx1 == *rowIdx2) //nonzeros at the same row index in both columns
        {
          // compute dot product between columns c1 and c2
          dot += alpha * (*value1) * (*value2);
          rowIdx1++; rowIdx2++;
          value1++; value2++; 
          newNonzero = true;   
        } else if (*rowIdx1 < *rowIdx2)
        {
          rowIdx1++; value1++;
        } else 
        {
          rowIdx2++; value2++;
        } 
      }

      // add nonzeros from beta*B, B is lower triangular NLP hessian
      if (nnz_idx_B < nonzeroes_B &&
          iRow_B[nnz_idx_B] == c1 &&
          jCol_B[nnz_idx_B] == c2)
      {
        dot += beta * values_B[nnz_idx_B];
        newNonzero = true;
        nnz_idx_B++;
      }

      // process the new nonzero element
      if (newNonzero)
      {
        //we need to use auxiliary storage
        //the actual sparse matrix is not assembled yet
        //because structure and nnz is unknown
        if (structureNotInitialized)
        {
          vvCols_C[c1].push_back(c2);
          vvValues_C[c1].push_back(dot);
        }
        //we can update directly #C.values
        else 
        {
          assert(_updateIterator < nonzeroes_C);
          assert(iRow_C[_updateIterator] == c1);
          assert(jCol_C[_updateIterator] == c2);
          values_C[_updateIterator] = dot;
          _updateIterator++;
        }
      } 
    }//end for c2
  }//end for c1

}

void hiopAugLagrHessian::appendScaledJacobian(hiopMatrixSparse &H, vector<vector<int>> &vvCols_H, vector<vector<double>> &vvValues_H, bool structureNotInitialized, double alpha, const hiopMatrixSparse &J, long long *cons_ineq_mapping)
{
 
  const int *iRow_J = J.get_iRow_const();  
  const int *jCol_J = J.get_jCol_const();  
  const double *values_J = J.get_values_const();  

  int *iRow_H = H.get_iRow();  
  int *jCol_H = H.get_jCol();  
  double *values_H = H.get_values();  

  //iterators
  int rowsAppended = 0;
  int ineq_i = 0;

  int previousRow = iRow_J[0];

  // iterate over rows/cols in J
  for (int i = 0; i < J.nnz(); i++)
  {
      const int row = iRow_J[i];
      
      //we have encountered new row in J
      if (previousRow != row)
      {
          if (i!= 0 && previousRow == cons_ineq_mapping[ineq_i]) rowsAppended++;
          if (i!= 0 && previousRow == cons_ineq_mapping[ineq_i]) ineq_i++;
          previousRow = row;
      }
  
      //append only Jacobian of inequality constraints (H_sx)
      if (row == cons_ineq_mapping[ineq_i])
      {
         if (structureNotInitialized)
         {
            vvCols_H[rowsAppended + nvars_nlp].push_back(jCol_J[i]);
            vvValues_H[rowsAppended + nvars_nlp].push_back(alpha * values_J[i]);
         }
         else
         {
            assert(iRow_H[_updateIterator] == rowsAppended + nvars_nlp);
            assert(jCol_H[_updateIterator] == jCol_J[i]);
            values_H[_updateIterator] = alpha * values_J[i];
            _updateIterator++;
         }
      
         //append scaled identity (H_ss) when reaching end of the row
         if ( i == J.nnz()-1 || row != iRow_J[i+1])
         {
           if (structureNotInitialized)
           {
              vvCols_H[rowsAppended + nvars_nlp].push_back(rowsAppended + nvars_nlp);
              vvValues_H[rowsAppended + nvars_nlp].push_back(-alpha);
           }
           else
           {
              assert(iRow_H[_updateIterator] == rowsAppended + nvars_nlp);
              assert(jCol_H[_updateIterator] == rowsAppended + nvars_nlp);
              values_H[_updateIterator] = -alpha;
              _updateIterator++;
           }
         }
      }

  }

  //std::cout << "\n\n\nineq = [";
  //for (int i = 0; i < nslacks_nlp; i++)
  //    std::cout << cons_ineq_mapping[i]+1 << ",";
  //std::cout << "];" << std::endl;

  assert(rowsAppended+1 == nslacks_nlp);
  assert(_updateIterator == _hessianAugLagr->nnz());
}

}

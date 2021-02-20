// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read �Additional BSD Notice� below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

/**
 * @file hiopNlpTransforms.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */

#ifndef HIOP_NLP_TRANSFORMS
#define HIOP_NLP_TRANSFORMS

#include "hiopInterface.hpp"
#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"

#include <cassert>
#include <list>
#include <vector>

namespace hiop
{

/** Template class for internal NLP manipulation/transformation.
 *
 * Examples of such transformations are removing fix variables, relaxing fixed 
 * variables, and problem rescaling.
 *
 * applyToXXX returns the transformation
 * applyInvToXXX returns the inverse transformation
 */
class hiopNlpTransformation
{
public:
  /* proxy for internal setup */
  virtual bool setup() = 0;

  /* number of vars in the NLP after the transformation */
  virtual long long n_post()=0;
  virtual long long n_post_local()=0;
  /* number of vars in the NLP to which the transformation is to be applied */
  virtual long long n_pre()=0;
  virtual long long n_pre_local()=0;

  /* transforms variable vector */
  virtual inline hiopVector* apply_to_x(hiopVector& x, const bool& new_x) { return &x; };
  virtual inline hiopVector* apply_inv_to_x(hiopVector& x) { return &x; }
  virtual inline void apply_inv_to_x(hiopVector& x_in, hiopVector& x_out) 
  { 
    //default implementation should have x_in as x_out's internal data array
    assert(x_in.local_data() == x_out.local_data());
  }

  virtual inline double apply_to_obj                  (double& f_in) { return f_in;} 
  virtual inline double apply_inv_to_obj              (double& f_in) { return f_in;} 

  virtual inline hiopVector* apply_to_grad_obj        (hiopVector& grad_in) { return &grad_in; }
  virtual inline hiopVector* apply_inv_to_grad_obj    (hiopVector& grad_in) { return &grad_in; }

  virtual inline hiopVector* apply_to_cons_eq         (hiopVector& c_in, const int& m_in) { return &c_in; }
  virtual inline hiopVector* apply_inv_to_cons_eq     (hiopVector& c_in, const int& m_in) { return &c_in; }
  virtual inline hiopVector* apply_to_cons_ineq       (hiopVector& c_in, const int& m_in) { return &c_in; }
  virtual inline hiopVector* apply_inv_to_cons_ineq   (hiopVector& c_in, const int& m_in) { return &c_in; }
  
  //the following two are for when the underlying NLP formulation works with full body constraints,
  //that is, evaluates both equalities and inequalities at once (a.k.a. one-call constraints and
  //and Jacobian evaluations)
  virtual inline hiopVector* apply_to_cons            (hiopVector&  cons_in, const int* m_in) { return &cons_in; }
  virtual inline hiopVector* apply_inv_to_cons        (hiopVector&  cons_in, const int* m_in) { return &cons_in; }

  virtual inline hiopMatrix* apply_to_jacob_eq        (hiopMatrix& Jac_in, const int& m_in) { return &Jac_in; }
  virtual inline hiopMatrix* apply_inv_to_jacob_eq    (hiopMatrix& Jac_in, const int& m_in) { return &Jac_in; }

  virtual inline hiopMatrix* apply_to_jacob_ineq      (hiopMatrix& Jac_in, const int& m_in) { return &Jac_in; }
  virtual inline hiopMatrix* apply_inv_to_jacob_ineq  (hiopMatrix& Jac_in, const int& m_in) { return &Jac_in; }  

  virtual inline hiopMatrix* apply_to_jacob_cons      (hiopMatrix& Jac_in, const int& m_in) { return &Jac_in; }
  virtual inline hiopMatrix* apply_inv_to_jacob_cons  (hiopMatrix& Jac_in, const int& m_in) { return &Jac_in; } 

  virtual inline hiopMatrix* apply_to_larg_hess       (hiopMatrix& Hess_in, const int& m_in) { return &Hess_in; }
  virtual inline hiopMatrix* apply_inv_to_larg_hess   (hiopMatrix& Hess_in, const int& m_in) { return &Hess_in; }
#if 0 //old code
  //! todo -> abstractize the below methods to work with other Jacobian types: sparse and MDS
  virtual inline double* apply_to_jacob_eq      (double* Jac_in, const int& m_in) { return Jac_in; }
  virtual inline double* apply_inv_to_jacob_eq   (double* Jac_in, const int& m_in) { return Jac_in; }
  
  virtual inline double* apply_to_jacob_ineq    (double* Jac_in, const int& m_in) { return Jac_in; }
  virtual inline double* apply_inv_to_jacob_ineq (double* Jac_in, const int& m_in) { return Jac_in; }
  
  //the following two are for when the underlying NLP formulation works with full body constraints
  virtual inline double* apply_to_jacob_cons    (double* Jac_in, const int& m_in) { return Jac_in; }
  virtual inline double* apply_inv_to_jacob_cons (double* Jac_in, const int& m_in) { return Jac_in; }

  //! todo -> transformations for Hessian ?!?
#endif // 0
  
public:
  hiopNlpTransformation(){};
  virtual ~hiopNlpTransformation() {};
};

/** Removes fixed variables from the NLP formulation.
 *
 * applyToXXX: takes the internal (reduced-space) XXX object (variables vector, function, gradient, etc) 
 * of the NLP formulation and returns it in the full-space (including the fixed 
 * variables) so that it is ready to be passed to user's calling code.
 *
 * applyInvToXXX: takes XXX as seen by the user calling code and returns the corresponding
 * reduced-space XXX object.
 */
class hiopFixedVarsRemover : public hiopNlpTransformation
{
public:
  hiopFixedVarsRemover(const hiopVector& xl,
                       const hiopVector& xu,
                       const double& fixedVarTol,
                       const long long& numFixedVars,
                       const long long& numFixedVars_local);
  ~hiopFixedVarsRemover();
public:
  /** inherited from the parent class */

  /* more setup methods (specific to this class) are defined below */
  virtual inline bool setup() {return true;}

  /* number of vars in the NLP after the tranformation */
  virtual inline long long n_post() { return fs_n(); }
  /* number of vars in the NLP to which the tranformation is to be applied */
  virtual inline long long n_pre () { return rs_n(); }

  virtual inline long long n_post_local() { return fs_n_local(); }
  virtual inline long long n_pre_local() { return rs_n_local(); }

  /* from reduced space to full space */
  inline hiopVector* apply_to_x(hiopVector& x, const bool& new_x) 
  { 
    x_rs_ref_ = &x;
    if(!new_x) { return x_fs; }
    apply_to_vector(&x, x_fs);
    return x_fs;
  };

  /* from full space to reduced space (fixed vars removed) */
  inline hiopVector* apply_inv_to_x(hiopVector& x_fs_in) 
  { 
    assert(x_rs_ref_!=NULL); assert(x_fs_in.local_data()==x_fs->local_data());
    apply_to_vector(&x_fs_in, x_rs_ref_);
    return x_rs_ref_; 
  }
  
  /* from fs to rs */
  inline void  apply_inv_to_x(hiopVector& x_in, hiopVector& xv_out)
  {
#ifdef HIOP_DEEPCHECKS
    assert(xv_out.get_size()<xl_fs->get_size());
#endif
    apply_inv_to_vector(&x_in, &xv_out);
  }
  
  /* from rs to fs and return the fs*/
  inline hiopVector* apply_to_grad_obj(hiopVector& grad_in)
  {
    grad_rs_ref = &grad_in;
    apply_to_vector(&grad_in, grad_fs);
    return grad_fs;
  }
  /* from fs to rs */
  inline hiopVector* apply_inv_to_grad_obj(hiopVector& grad_in)
  {
    assert(&grad_in==grad_fs);
    apply_inv_to_vector(&grad_in, grad_rs_ref);
    return grad_rs_ref;
  }
  /* from rs to fs */
  inline hiopMatrix* apply_to_jacob_eq(hiopMatrix& Jac_in, const int& m_in)
  {
    hiopMatrixDense* Jac_de = dynamic_cast<hiopMatrixDense*>(&Jac_in);
    if(Jac_de==nullptr) {
      return nullptr;
    }
    Jacc_rs_ref = Jac_de;
    assert(Jacc_fs->m()==m_in);
    applyToMatrix(Jac_de->local_data(), m_in, Jacc_fs->local_data());
    return Jacc_fs;
  }
  inline hiopMatrix* apply_inv_to_jacob_eq(hiopMatrix&  Jac_in, const int& m_in)
  {
    hiopMatrixDense* Jac_de = dynamic_cast<hiopMatrixDense*>(&Jac_in);
    if(Jac_de==NULL) {
      return nullptr;
    }    
    assert(Jacc_fs->m()==m_in);
    applyInvToMatrix(Jac_de->local_data(), m_in, Jacc_rs_ref->local_data());
    return Jacc_rs_ref;
  }
  inline hiopMatrix* apply_to_jacob_ineq(hiopMatrix& Jac_in, const int& m_in)
  {
    hiopMatrixDense* Jac_de = dynamic_cast<hiopMatrixDense*>(&Jac_in);
    if(Jac_de==NULL) {
      return nullptr;
    }
    Jacd_rs_ref = Jac_de;
    assert(Jacd_fs->m()==m_in);
    applyToMatrix(Jac_de->local_data(), m_in, Jacd_fs->local_data());
    return Jacd_fs;    
  }
  inline hiopMatrix* apply_inv_to_jacob_ineq(hiopMatrix& Jac_in, const int& m_in)
  {
    hiopMatrixDense* Jac_de = dynamic_cast<hiopMatrixDense*>(&Jac_in);
    if(Jac_de==NULL) {
      return nullptr;
    }    
    assert(Jacd_fs->m()==m_in);
    applyInvToMatrix(Jac_de->local_data(), m_in, Jacd_rs_ref->local_data());
    return Jacd_rs_ref;
  }

  /** methods not inherited from parent class */
  bool setupDecisionVectorPart();
  bool setupConstraintsPart(const int& neq, const int& nineq);
#ifdef HIOP_USE_MPI
  /* saves the inter-process distribution of (primal) vectors distribution */
  void setFSVectorDistrib(long long* vec_distrib,int num_ranks);
  /* allocates and returns the reduced-space column partitioning to be used internally by HiOp */
  long long* allocRSVectorDistrib();
  inline void setMPIComm(const MPI_Comm& commIn) { comm = commIn; }
#endif
  /* "copies" a full space vector to a reduced space vector */
  void copyFsToRs(const hiopVector& fsVec,  hiopVector& rsVec);
  void copyFsToRs(const hiopInterfaceBase::NonlinearityType* fs, hiopInterfaceBase::NonlinearityType* rs);
  
  inline long long fs_n() const { return n_fs;}
  inline long long rs_n() const { return n_rs;}
  inline long long fs_n_local() const { assert(xl_fs); return xl_fs->get_local_size(); }
  inline long long rs_n_local() const { assert(xl_fs); return fs_n_local()-n_fixed_vars_local;}
protected: 
#if 0 //old interface
  void applyToArray   (const double* vec_rs, double* vec_fs);
  void applyInvToArray(const double* vec_fs, double* vec_rs);
#endif
  void apply_to_vector   (const hiopVector* vec_rs, hiopVector* vec_fs);
  void apply_inv_to_vector(const hiopVector* vec_fs, hiopVector* vec_rs);
  
  void applyToMatrix   (const double* M_rs, const int& m_in, double* M_fs);
  void applyInvToMatrix(const double* M_fs, const int& m_in, double* M_rs);
protected:
  long long n_fixed_vars_local;
  long long n_fixed_vars;

  double fixedVarTol;

  long long n_fs; //full-space n
  long long n_rs; //reduced-space n

  //working buffer used to hold the full-space (user's) vector of decision variables and other optimiz objects
  hiopVector*x_fs, *grad_fs;
  //working buffers for the full-space Jacobians
  hiopMatrixDense *Jacc_fs, *Jacd_fs;
  
  hiopMatrixDense *Jacc_rs_ref;
  hiopMatrixDense *Jacd_rs_ref;

  //a copy of the lower and upper bounds provided by user
  hiopVector*xl_fs, *xu_fs;
  //indexes corresponding to fixed variables (local indexes)
  std::vector<int> fs2rs_idx_map;

  //references to reduced-space buffers - returned in applyInvXXX
  hiopVector* x_rs_ref_;
  hiopVector* grad_rs_ref;
#ifdef HIOP_USE_MPI
  std::vector<long long> fs_vec_distrib;
  MPI_Comm comm;
#endif
  
};

class hiopFixedVarsRelaxer : public hiopNlpTransformation
{
public: 
  hiopFixedVarsRelaxer(const hiopVector& xl,
                       const hiopVector& xu,
                       const long long& numFixedVars,
                       const long long& numFixedVars_local);
  virtual ~hiopFixedVarsRelaxer();

  /* number of vars in the NLP after the tranformation */
  inline long long n_post()  { /*assert(xl_copy);*/ return n_vars; } //xl_copy->get_size(); }
  /* number of vars in the NLP to which the tranformation is to be applied */
  virtual long long n_pre () { /*assert(xl_copy);*/ return n_vars; } //xl_copy->get_size(); }

  inline long long n_post_local()  { return n_vars_local; } //xl_copy->get_local_size(); }
  inline long long n_pre_local()  { return n_vars_local; } //xl_copy->get_local_size(); }

  inline bool setup() { return true; }

  void relax(const double& fixed_var_tol, const double& fixed_var_perturb, 
	     hiopVector& xl, hiopVector& xu);
private:
  hiopVector*xl_copy, *xu_copy;
  long long  n_vars; int n_vars_local;
};

/** 
 * @brief Scale the NLP formulation before solving the problem
 *
 * scale the NLP objective using the maximum gradient approach.
 * scale the NLP constraints using the maximum gradient approach.
 */
class hiopNLPObjGradScaling : public hiopNlpTransformation
{
public:
  hiopNLPObjGradScaling(const double max_grad, hiopVector* gradf);
  ~hiopNLPObjGradScaling();
public:
  /** inherited from the parent class */

  /* more setup methods (specific to this class) are defined below */
  virtual inline bool setup() {return true;}

  /* number of vars in the NLP after the tranformation */
  inline long long n_post()  { return n_vars; } 
  /* number of vars in the NLP to which the tranformation is to be applied */
  virtual long long n_pre () { return n_vars; } 

  inline long long n_post_local()  { return n_vars_local; }
  inline long long n_pre_local()  { return n_vars_local; }

  /* from scaled to unscaled objective*/
  inline double apply_to_obj(double& f_in) { return f_in/obj_scale;}
  /* from unscaled to scaled objective*/
  inline double apply_inv_to_obj(double& f_in) { return obj_scale*f_in;}
  
  /// @brief return the scaling fact for objective
  inline double get_obj_scale() const {return obj_scale;}

  /* from scaled to unscaled*/
  inline hiopVector* apply_to_grad_obj(hiopVector& grad_in)
  {
    grad_in.scale(1./obj_scale);
    return &grad_in;
  }
  
  /* from unscaled to scaled*/
  inline hiopVector* apply_inv_to_grad_obj(hiopVector& grad_in)
  {
    grad_in.scale(obj_scale);
    return &grad_in;
  }


  
#if 0
  /* from rs to fs */
  inline double* apply_to_jacob_eq(double* Jac_in, const int& m_in)
  {
    Jacc_rs_ref = Jac_in;
    assert(Jacc_fs->m()==m_in);
    //applyToMatrix(Jac_in, m_in, Jacc_fs->get_M());
    applyToMatrix(Jac_in, m_in, Jacc_fs->local_data());
    return Jacc_fs->local_data();
  }
  inline double* apply_inv_to_jacob_eq(double* Jac_in, const int& m_in)
  {
    assert(Jacc_fs->m()==m_in);
    applyInvToMatrix(Jac_in, m_in, Jacc_rs_ref);
    return Jacc_rs_ref;
  }
  inline double* apply_to_jacob_ineq(double* Jac_in, const int& m_in)
  {
    Jacd_rs_ref = Jac_in;
    assert(Jacd_fs->m()==m_in);
    //applyToMatrix(Jac_in, m_in, Jacd_fs->get_M());
    applyToMatrix(Jac_in, m_in, Jacd_fs->local_data());
    return Jacd_fs->local_data();
  }
  inline double* apply_inv_to_jacob_ineq(double* Jac_in, const int& m_in)
  {
    assert(Jacd_fs->m()==m_in);
    applyInvToMatrix(Jac_in, m_in, Jacd_rs_ref);
    return Jacd_rs_ref;
  }


  
protected: 
  void applyToArray   (const double* vec_rs, double* vec_fs);
  void applyInvToArray(const double* vec_fs, double* vec_rs);

  void applyToMatrix   (const double* M_rs, const int& m_in, double* M_fs);
  void applyInvToMatrix(const double* M_fs, const int& m_in, double* M_rs);
#endif  

private:
  long long n_vars, n_vars_local;
  double obj_scale;

#if 0
  hiopVector *scal_jacc, *scal_jacd;
  hiopMatrix *Jacc_scaled, *Jacd_scaled;
  hiopMatrix *Jacc_unscaled, *Jacd_unscaled;
  hiopMatrix *Hess_scaled;
  hiopMatrix *Hess_unscaled;
#endif // 0
};



class hiopNlpTransformations : public hiopNlpTransformation
{
public:
  hiopNlpTransformations() : n_vars_usernlp(-1), n_vars_local_usernlp(-1) { };
  virtual ~hiopNlpTransformations() 
  {
    std::list<hiopNlpTransformation*>::iterator it;
    for(it=list_trans_.begin(); it!=list_trans_.end(); it++)
      delete (*it);
  };

  inline bool setup() { return true; }
  inline void setUserNlpNumVars(const long long& n_vars) { n_vars_usernlp = n_vars; }
  inline void setUserNlpNumLocalVars(const long long& n_vars) { n_vars_local_usernlp = n_vars; }
  inline void append(hiopNlpTransformation* t) { list_trans_.push_back(t); }
  inline void clear() { 
    std::list<hiopNlpTransformation*>::iterator it;
    for(it=list_trans_.begin(); it!=list_trans_.end(); it++)
      delete (*it);
    list_trans_.clear(); 
  }

  /* number of vars in the NLP after the tranformation */
  inline virtual long long n_post() 
  { 
#ifdef HIOP_DEEPCHECKS
      assert(n_vars_usernlp>0);
#endif 
    if(list_trans_.empty()) {
      return n_vars_usernlp;
    } else {
#ifdef HIOP_DEEPCHECKS
      assert(n_vars_usernlp==list_trans_.back()->n_post());
#endif 
      return list_trans_.back()->n_post(); 
    }
  }
  inline virtual long long n_post_local() 
  { 
#ifdef HIOP_DEEPCHECKS
      assert(n_vars_usernlp>0);
#endif 
    if(list_trans_.empty()) {
      return n_vars_local_usernlp;
    } else {
#ifdef HIOP_DEEPCHECKS
      assert(n_vars_usernlp==list_trans_.back()->n_post());
      assert(n_vars_local_usernlp==list_trans_.back()->n_post_local());
#endif 
      return list_trans_.back()->n_post_local(); 
    }
  }
  /* number of vars in the NLP to which the tranformation is to be applied */
  inline virtual long long n_pre() 
  { 
#ifdef HIOP_DEEPCHECKS
      assert(n_vars_usernlp>0);
#endif 
    if(list_trans_.empty()) {
      return n_vars_usernlp;
    } else {
#ifdef HIOP_DEEPCHECKS
      assert(n_vars_usernlp==list_trans_.back()->n_post());
#endif
      return list_trans_.front()->n_pre(); 
    }
  }
  /* number of local vars in the NLP to which the tranformation is to be applied */
  inline virtual long long n_pre_local() 
  { 
#ifdef HIOP_DEEPCHECKS
    assert(n_vars_usernlp>0);
#endif 
    if(list_trans_.empty()) {
      return n_vars_local_usernlp;
    } else {
#ifdef HIOP_DEEPCHECKS
      assert(n_vars_usernlp==list_trans_.back()->n_post());
#endif
      return list_trans_.front()->n_pre_local(); 
    }
  }
  
  hiopVector* apply_to_x(hiopVector& x, const bool& new_x) 
  {
    hiopVector* ret = &x;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->apply_to_x(*ret ,new_x);
    return ret;
  }

  virtual hiopVector* apply_inv_to_x(hiopVector& x)
  { 
    assert(false && "This overload of apply_inv_to_x is not implemented in hiopNlpTransformations class\n");
    return nullptr;
  }

  void apply_inv_to_x(hiopVector& x_in, hiopVector& x_out) 
  {
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it) {
      (*it)->apply_inv_to_x(x_in, x_out);
    }
  }

  double apply_to_obj(double& f_in) 
  {
    double ret = f_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->apply_to_obj(ret);
    return ret;
  }

  double apply_inv_to_obj(double& f_in) 
  {
    double ret = f_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->apply_inv_to_obj(ret);
    return ret;
  }  
  
  hiopVector* apply_to_grad_obj(hiopVector& grad_in) 
  {
    hiopVector* ret = &grad_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->apply_to_grad_obj(*ret);
    return ret;
  }

  hiopVector* apply_inv_to_grad_obj(hiopVector& grad_in) 
  {
    hiopVector* ret = &grad_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->apply_inv_to_grad_obj(*ret);
    return ret;
  }

  hiopMatrix* apply_to_jacob_eq(hiopMatrix& Jac_in, const int& m_in)
  {
    hiopMatrix* ret = &Jac_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->apply_to_jacob_eq(*ret, m_in);
    return ret;
  }

  hiopMatrix* apply_inv_to_jacob_eq(hiopMatrix& Jac_in, const int& m_in)
  {
    hiopMatrix* ret = &Jac_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->apply_inv_to_jacob_eq(*ret, m_in);
    return ret;
  }

  hiopMatrix* apply_to_jacob_ineq(hiopMatrix& Jac_in, const int& m_in)
  {
    hiopMatrix* ret = &Jac_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->apply_to_jacob_ineq(*ret, m_in);
    return ret;
  }

  hiopMatrix* apply_inv_to_jacob_ineq(hiopMatrix& Jac_in, const int& m_in)
  {
    hiopMatrix* ret = &Jac_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->apply_inv_to_jacob_ineq(*ret, m_in);
    return ret;
  }


private:
  std::list<hiopNlpTransformation*> list_trans_;
  long long  n_vars_usernlp, n_vars_local_usernlp;
};

}
#endif

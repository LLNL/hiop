// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
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
  virtual inline hiopVector* applyTox(hiopVector& x, const bool& new_x) { return &x; };
  virtual inline hiopVector* applyInvTox(hiopVector& x) { return &x; }
  virtual inline void applyInvTox(hiopVector& x_in, hiopVector& x_out) 
  { 
    //default implementation should have x_in as x_out's internal data array
    assert(x_in.local_data() == x_out.local_data());
  }

  virtual inline double applyToObj(double& f_in) { return f_in;} 
  virtual inline double applyInvToObj(double& f_in) { return f_in;} 

  virtual inline double* applyToGradObj(double* grad_in) { return grad_in; }
  virtual inline double* applyInvToGradObj(double* grad_in) { return grad_in; }

  virtual inline double* applyToConsEq(double* c_in, const int& m_in) { return c_in; }
  virtual inline double* applyInvToConsEq(double* c_in, const int& m_in) { return c_in; }
  virtual inline double* applyToConsIneq(double* c_in, const int& m_in) { return c_in; }
  virtual inline double* applyInvToConsIneq(double* c_in, const int& m_in) { return c_in; }
  //the following two are for when the underlying NLP formulation works with full body constraints,
  //that is, evaluates both equalities and inequalities at once (a.k.a. one-call constraints and
  //and Jacobian evaluations)
  virtual inline double* applyToCons(double* cons_in, const int* m_in) { return cons_in; }
  virtual inline double* applyInvToCons(double* cons_in, const int* m_in) { return cons_in; }

  //! todo -> abstractize the below methods to work with other Jacobian types: sparse and MDS
  virtual inline double* applyToJacobEq      (double* Jac_in, const int& m_in) { return Jac_in; }
  virtual inline double* applyInvToJacobEq   (double* Jac_in, const int& m_in) { return Jac_in; }
  virtual inline double* applyToJacobIneq    (double* Jac_in, const int& m_in) { return Jac_in; }
  virtual inline double* applyInvToJacobIneq (double* Jac_in, const int& m_in) { return Jac_in; }
  virtual inline double* applyToJacobCons    (double* Jac_in, const int& m_in) { return Jac_in; }
  //the following two are for when the underlying NLP formulation works with full body constraints
  virtual inline double* applyInvToJacobCons (double* Jac_in, const int& m_in) { return Jac_in; }

  //! todo -> transformations for Hessian ?!?
public:
  hiopNlpTransformation() {}; 
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
  inline hiopVector* applyTox(hiopVector& x, const bool& new_x) 
  { 
    x_rs_ref_ = &x;
    if(!new_x) { return x_fs; }
    applyToArray(x.local_data_const(), x_fs->local_data());
    return x_fs;
  };

  /* from full space to reduced space (fixed vars removed) */
  inline hiopVector* applyInvTox(hiopVector& x_fs_in) 
  { 
    assert(x_rs_ref_!=NULL); assert(x_fs_in.local_data()==x_fs->local_data());
    applyInvToArray(x_fs_in.local_data(), x_rs_ref_->local_data());
    return x_rs_ref_; 
  }
  
  /* from fs to rs */
  inline void  applyInvTox(hiopVector& x_in, hiopVector& xv_out)
  {
#ifdef HIOP_DEEPCHECKS
    assert(xv_out.get_size()<xl_fs->get_size());
#endif
    applyInvToArray(x_in.local_data(), xv_out.local_data());
  }
  
  /* from rs to fs and return the fs*/
  inline double* applyToGradObj(double* grad_in)
  {
    grad_rs_ref = grad_in;
    applyToArray(grad_in, grad_fs->local_data());
    return grad_fs->local_data();
  }
  /* from fs to rs */
  inline double* applyInvToGradObj(double* grad_in)
  {
    assert(grad_in==grad_fs->local_data());
    applyInvToArray(grad_in, grad_rs_ref);
    return grad_rs_ref;
  }
  /* from rs to fs */
  inline double* applyToJacobEq(double* Jac_in, const int& m_in)
  {
    Jacc_rs_ref = Jac_in;
    assert(Jacc_fs->m()==m_in);
    //applyToMatrix(Jac_in, m_in, Jacc_fs->get_M());
    applyToMatrix(Jac_in, m_in, Jacc_fs->local_data());
    return Jacc_fs->local_data();
  }
  inline double* applyInvToJacobEq(double* Jac_in, const int& m_in)
  {
    assert(Jacc_fs->m()==m_in);
    applyInvToMatrix(Jac_in, m_in, Jacc_rs_ref);
    return Jacc_rs_ref;
  }
  inline double* applyToJacobIneq(double* Jac_in, const int& m_in)
  {
    Jacd_rs_ref = Jac_in;
    assert(Jacd_fs->m()==m_in);
    //applyToMatrix(Jac_in, m_in, Jacd_fs->get_M());
    applyToMatrix(Jac_in, m_in, Jacd_fs->local_data());
    return Jacd_fs->local_data();
  }
  inline double* applyInvToJacobIneq(double* Jac_in, const int& m_in)
  {
    assert(Jacd_fs->m()==m_in);
    applyInvToMatrix(Jac_in, m_in, Jacd_rs_ref);
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
  void applyToArray   (const double* vec_rs, double* vec_fs);
  void applyInvToArray(const double* vec_fs, double* vec_rs);

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

  //a copy of the lower and upper bounds provided by user
  hiopVector*xl_fs, *xu_fs;
  //indexes corresponding to fixed variables (local indexes)
  std::vector<int> fs2rs_idx_map;

  //references to reduced-space buffers - returned in applyInvXXX
  hiopVector* x_rs_ref_;
  double* grad_rs_ref;
  double *Jacc_rs_ref, *Jacd_rs_ref;
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

/** Scale the NLP formulation.
 *
 * scale the NLP objective using the maximum gradient information.
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

  /* from scaled to unscaled*/
  void applyToArray(const double* vec_s, double* vec_us);
  /* from unscaled to scaled*/
  void applyInvToArray(const double* vec_s, double* vec_us);  
  
  /* from scaled to unscaled objective*/
  inline double applyToObj(double& f_in) { return f_in/obj_scale;}
  /* from unscaled to scaled objective*/
  inline double applyInvToObj(double& f_in) { return obj_scale*f_in;}
  
  inline double get_obj_scale() const {return obj_scale;}

#if 1 //old interface
  /* from scaled to unscaled*/
  inline double* applyToGradObj(double* grad_in)
  {
    grad_scaled_ref = grad_in;
    applyToArray(grad_in, grad_unscaled->local_data());
    return grad_unscaled->local_data();
  }
  /* from unscaled to scaled*/
  inline double* applyInvToGradObj(double* grad_in)
  {
    assert(grad_in==grad_unscaled->local_data());
    applyInvToArray(grad_in, grad_scaled_ref);
    return grad_scaled_ref;
  }
#else
  /* from scaled to unscaled*/
  hiopVector* applyToGradObj(hiopVector* grad_in);
  /* from unscaled to scaled*/
  hiopVector* applyInvToGradObj(hiopVector* grad_in);
#endif
  
  
#if 0
  /* from rs to fs */
  inline double* applyToJacobEq(double* Jac_in, const int& m_in)
  {
    Jacc_rs_ref = Jac_in;
    assert(Jacc_fs->m()==m_in);
    //applyToMatrix(Jac_in, m_in, Jacc_fs->get_M());
    applyToMatrix(Jac_in, m_in, Jacc_fs->local_data());
    return Jacc_fs->local_data();
  }
  inline double* applyInvToJacobEq(double* Jac_in, const int& m_in)
  {
    assert(Jacc_fs->m()==m_in);
    applyInvToMatrix(Jac_in, m_in, Jacc_rs_ref);
    return Jacc_rs_ref;
  }
  inline double* applyToJacobIneq(double* Jac_in, const int& m_in)
  {
    Jacd_rs_ref = Jac_in;
    assert(Jacd_fs->m()==m_in);
    //applyToMatrix(Jac_in, m_in, Jacd_fs->get_M());
    applyToMatrix(Jac_in, m_in, Jacd_fs->local_data());
    return Jacd_fs->local_data();
  }
  inline double* applyInvToJacobIneq(double* Jac_in, const int& m_in)
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
  
  hiopVector *grad_scaled;
  hiopVector *grad_unscaled;
  double* grad_scaled_ref;
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
  
  hiopVector* applyTox(hiopVector& x, const bool& new_x) 
  {
    hiopVector* ret = &x;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->applyTox(*ret ,new_x);
    return ret;
  }

  virtual hiopVector* applyInvTox(hiopVector& x)
  { 
    assert(false && "This overload of applyInvTox is not implemented in hiopNlpTransformations class\n");
    return nullptr;
  }

  void applyInvTox(hiopVector& x_in, hiopVector& x_out) 
  {
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it) {
      (*it)->applyInvTox(x_in, x_out);
    }
  }

  double applyToObj(double& f_in) 
  {
    double ret = f_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->applyToObj(ret);
    return ret;
  }

  double applyInvToObj(double& f_in) 
  {
    double ret = f_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->applyInvToObj(ret);
    return ret;
  }  
  
  double* applyToGradObj(double* grad_in) 
  {
    double* ret = grad_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->applyToGradObj(ret);
    return ret;
  }

  double* applyInvToGradObj(double* grad_in) 
  {
    double* ret = grad_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->applyInvToGradObj(ret);
    return ret;
  }

  double* applyToJacobEq(double* Jac_in, const int& m_in)
  {
    double* ret = Jac_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->applyToJacobEq(ret, m_in);
    return ret;
  }

  double* applyInvToJacobEq(double* Jac_in, const int& m_in)
  {
    double* ret = Jac_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->applyInvToJacobEq(ret, m_in);
    return ret;
  }

  double* applyToJacobIneq(double* Jac_in, const int& m_in)
  {
    double* ret = Jac_in;
    for(std::list<hiopNlpTransformation*>::iterator it=list_trans_.begin(); it!=list_trans_.end(); ++it)
      ret = (*it)->applyToJacobIneq(ret, m_in);
    return ret;
  }

  double* applyInvToJacobIneq(double* Jac_in, const int& m_in)
  {
    double* ret = Jac_in;
    for(std::list<hiopNlpTransformation*>::reverse_iterator it=list_trans_.rbegin(); it!=list_trans_.rend(); ++it)
      ret = (*it)->applyInvToJacobIneq(ret, m_in);
    return ret;
  }


private:
  std::list<hiopNlpTransformation*> list_trans_;
  long long  n_vars_usernlp, n_vars_local_usernlp;
};

}
#endif

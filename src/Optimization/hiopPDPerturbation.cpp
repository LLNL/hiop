// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read "Additional BSD Notice" below.
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
 * @file hiopPDPerturbation.cpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
 
#include "hiopPDPerturbation.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace hiop
{

  /** Initializes and reinitializes object based on the 'options' parameters of the
   * 'nlp_' object.
   * Returns 'false' if something goes wrong, otherwise 'true'
   */
  bool hiopPDPerturbationBase::initialize(hiopNlpFormulation* nlp)
  {
    nlp_ = nlp;
    delta_w_min_bar_ = nlp->options->GetNumeric("delta_w_min_bar");
    delta_w_max_bar_ = nlp->options->GetNumeric("delta_w_max_bar");
    delta_w_0_bar_   = nlp->options->GetNumeric("delta_0_bar");
    kappa_w_minus_   = nlp->options->GetNumeric("kappa_w_minus");
    kappa_w_plus_bar_= nlp->options->GetNumeric("kappa_w_plus_bar");
    kappa_w_plus_    = nlp->options->GetNumeric("kappa_w_plus");
    delta_c_bar_     = nlp->options->GetNumeric("delta_c_bar");
    kappa_c_         = nlp->options->GetNumeric("kappa_c");

    delta_wx_curr_ = nlp_->alloc_primal_vec();
    delta_wd_curr_ = nlp_->alloc_dual_ineq_vec();
    delta_cc_curr_ = nlp_->alloc_dual_eq_vec();
    delta_cd_curr_ = nlp_->alloc_dual_ineq_vec();
    delta_wx_last_ = nlp_->alloc_primal_vec();
    delta_wd_last_ = nlp_->alloc_dual_ineq_vec();
    delta_cc_last_ = nlp_->alloc_dual_eq_vec();
    delta_cd_last_ = nlp_->alloc_dual_ineq_vec();

    delta_wx_curr_->setToZero();
    delta_wd_curr_->setToZero();
    delta_cc_curr_->setToZero();
    delta_cd_curr_->setToZero();
    delta_wx_last_->setToZero();
    delta_wd_last_->setToZero();
    delta_cc_last_->setToZero();
    delta_cd_last_->setToZero();

    num_degen_iters_ = 0;

    deltas_test_type_ = dttNoTest;
    return true;
  }

  /** Decides degeneracy @hess_degenerate_ and @jac_degenerate_ based on @deltas_test_type_ 
   *  when the @num_degen_iters_ > @num_degen_max_iters_
   */
  void hiopPDPerturbationBase::update_degeneracy_type()
  {
   switch (deltas_test_type_) {
   case dttNoTest:
     return;
   case dttDeltac0Deltaw0:
     if(hess_degenerate_ == dtNotEstablished &&
	     jac_degenerate_ == dtNotEstablished) {
       hess_degenerate_ = dtNotDegenerate;
       jac_degenerate_ = dtNotDegenerate;
     } else if(hess_degenerate_ == dtNotEstablished) {
       hess_degenerate_ = dtNotDegenerate;
     } else if(jac_degenerate_ == dtNotEstablished) {
       jac_degenerate_ = dtNotDegenerate;
     }
     break;
   case dttDeltacposDeltaw0:
     if(hess_degenerate_ == dtNotEstablished) {
       hess_degenerate_ = dtNotDegenerate;
     }
     
     if(jac_degenerate_ == dtNotEstablished) {
       num_degen_iters_++;
       if(num_degen_iters_ >= num_degen_max_iters_) {
	       jac_degenerate_ = dtDegenerate;
       }
     }
     break;
   case dttDeltac0Deltawpos:
     if(jac_degenerate_ == dtNotEstablished) {
        jac_degenerate_ = dtNotDegenerate;
	
     }
     if(hess_degenerate_ == dtNotEstablished) {
       num_degen_iters_++;
       if(num_degen_iters_ >= num_degen_max_iters_) {
         hess_degenerate_ = dtDegenerate;
       }
     }
     break;
   case dttDeltacposDeltawpos:
     num_degen_iters_++;
     if(num_degen_iters_ >= num_degen_max_iters_) {
       hess_degenerate_ = dtDegenerate;
       jac_degenerate_ = dtDegenerate;
     }
     break;
   }
  }

  hiopPDPerturbation::hiopPDPerturbation()
    : hiopPDPerturbationBase(),
      delta_wx_curr_db_{0.},
      delta_wd_curr_db_{0.},
      delta_cc_curr_db_{0.},
      delta_cd_curr_db_{0.},
      delta_wx_last_db_{0.},
      delta_wd_last_db_{0.},
      delta_cc_last_db_{0.},
      delta_cd_last_db_{0.}
  {
  }

  hiopPDPerturbation::~hiopPDPerturbation()
  {
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  bool hiopPDPerturbation::compute_initial_deltas(hiopVector& delta_wx,
                                                  hiopVector& delta_wd,
                                                  hiopVector& delta_cc,
                                                  hiopVector& delta_cd)
  {
    double delta_temp;
    double delta_temp2;
    update_degeneracy_type();

    if(delta_wx_curr_db_>0.) {
      delta_wx_last_db_ = delta_wx_curr_db_;      
    }
    if(delta_wd_curr_db_>0.) {
      delta_wd_last_db_ = delta_wd_curr_db_;
    }
    if(delta_cc_curr_db_>0.) {
      delta_cc_last_db_ = delta_cc_curr_db_;      
    }
    if(delta_cd_curr_db_>0.) {
      delta_cd_last_db_= delta_cd_curr_db_;
    }

    set_delta_last_vec();

    if(hess_degenerate_ == dtNotEstablished || jac_degenerate_ == dtNotEstablished) {
      deltas_test_type_ = dttDeltac0Deltaw0;
    } else {
      deltas_test_type_ = dttNoTest;
    }

    if(jac_degenerate_ == dtDegenerate) {
      delta_temp = compute_delta_c(mu_);
    } else {
      delta_temp = 0.0;
    }
    delta_cc.setToConstant(delta_temp);
    delta_cd.setToConstant(delta_temp);
    delta_cc_curr_db_ = delta_temp;
    delta_cd_curr_db_ = delta_temp;

    if(hess_degenerate_ == dtDegenerate) {
      delta_wx_curr_db_ = 0.;
      delta_wd_curr_db_ = 0.;
      if(!guts_of_compute_perturb_wrong_inertia(delta_temp, delta_temp2)) {
        return false;
      }
    } else {
      delta_temp = delta_temp2 = 0.;
    }
    delta_wx.setToConstant(delta_temp);
    delta_wd.setToConstant(delta_temp2);
    delta_wx_curr_db_ = delta_temp;
    delta_wd_curr_db_ = delta_temp2;

    set_delta_curr_vec();

    return true;
  }

  /** Method for correcting inertia */
  bool hiopPDPerturbation::compute_perturb_wrong_inertia(hiopVector& delta_wx,
                                                         hiopVector& delta_wd,
                                                         hiopVector& delta_cc,
                                                         hiopVector& delta_cd)
  {    
    update_degeneracy_type();

    assert(delta_wx_curr_db_ == delta_wd_curr_db_);
    assert(delta_cc_curr_db_ == delta_cd_curr_db_);
    
    double delta_wx_temp{0.0};
    double delta_wd_temp{0.0};
    
    bool ret = guts_of_compute_perturb_wrong_inertia(delta_wx_temp, delta_wd_temp);
    if(!ret && delta_cc_curr_db_ == 0.) {
      delta_wx_curr_db_ = delta_wd_curr_db_ = 0.;
      delta_cc_curr_db_ = delta_cd_curr_db_ = compute_delta_c(mu_);      
      deltas_test_type_ = dttNoTest;
      if(hess_degenerate_ == dtDegenerate) {
        hess_degenerate_ = dtNotEstablished;
      }

      ret = guts_of_compute_perturb_wrong_inertia(delta_wx_temp, delta_wd_temp);
    }

    delta_cc.setToConstant(delta_cc_curr_db_);
    delta_cd.setToConstant(delta_cd_curr_db_);
    delta_wx.setToConstant(delta_wx_temp);
    delta_wd.setToConstant(delta_wd_temp);
    set_delta_curr_vec();
    return ret;
  }

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  bool hiopPDPerturbation::compute_perturb_singularity(hiopVector& delta_wx, 
                                                       hiopVector& delta_wd,
                                                       hiopVector& delta_cc,
                                                       hiopVector& delta_cd)
  {    
    assert(delta_wx_curr_db_ == delta_wd_curr_db_);
    assert(delta_cc_curr_db_ == delta_cd_curr_db_);
    double delta_wx_temp{0.0};
    double delta_wd_temp{0.0};
    
    if (hess_degenerate_ == dtNotEstablished ||
        jac_degenerate_ == dtNotEstablished) {
      switch (deltas_test_type_) {
      case dttDeltac0Deltaw0:
        //this is the first call
        if (jac_degenerate_ == dtNotEstablished) {
          delta_cc_curr_db_ = delta_cd_curr_db_ = compute_delta_c(mu_);
          deltas_test_type_ = dttDeltacposDeltaw0;
        }
        else {
          assert(hess_degenerate_ == dtNotEstablished);
          if(!guts_of_compute_perturb_wrong_inertia(delta_wx_temp, delta_wd_temp)) {
            return false;            
          }
          assert(delta_cc.is_zero() && delta_cd.is_zero());
          deltas_test_type_ = dttDeltac0Deltawpos;
        }
        break;
      case dttDeltacposDeltaw0:
        assert(delta_wx_curr_db_ == 0. && delta_cc_curr_db_ > 0.);
        assert(jac_degenerate_ == dtNotEstablished);
        delta_cd_curr_db_ = delta_cc_curr_db_ = 0.;
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx_temp, delta_wd_temp)) {
          return false;
        }
        deltas_test_type_ = dttDeltac0Deltawpos;
        break;
      case dttDeltac0Deltawpos:
        assert(delta_wx_curr_db_ > 0. && delta_cc_curr_db_ == 0.);
        delta_cc_curr_db_ = delta_cd_curr_db_ = compute_delta_c(mu_);
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx_temp, delta_wd_temp)) {
          return false;
        }
        deltas_test_type_ = dttDeltacposDeltawpos;
        break;
      case dttDeltacposDeltawpos:
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx_temp, delta_wd_temp))
          return false;
        break;
      case dttNoTest:
        assert(false && "something went wrong - should not get here");
      }
    } else {
      if(delta_cc_curr_db_ > 0.) { 
        // If we already used a perturbation for the constraints, we do
        // the same thing as if we were encountering negative curvature
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx_temp, delta_wd_temp)) {
          //todo: need some error message (so that we know what and more important
          //where something went wrong)
          return false;
        }
      } else {
        // Otherwise we now perturb the Jacobian part 
        delta_cd_curr_db_ = delta_cc_curr_db_ = compute_delta_c(mu_);
      }
    }
  
    delta_wx.setToConstant(delta_wx_curr_db_);
    delta_wd.setToConstant(delta_wd_curr_db_);
    delta_cc.setToConstant(delta_cc_curr_db_);
    delta_cd.setToConstant(delta_cd_curr_db_);
  
    set_delta_curr_vec();

    return true;
  }

  /** 
   * Internal method implementing the computation of delta_w's to correct wrong inertia
   */
  bool hiopPDPerturbation::guts_of_compute_perturb_wrong_inertia(double& delta_wx, double& delta_wd)
  {
    assert(delta_wx_curr_db_ == delta_wd_curr_db_ && "these should be equal");
    assert(delta_wx_last_db_ == delta_wd_last_db_ && "these should be equal");
    if(delta_wx_curr_db_ == 0.) {
      if(delta_wx_last_db_ == 0.) {
        delta_wx_curr_db_ = delta_w_0_bar_;
      } else {
        delta_wx_curr_db_ = std::fmax(delta_w_min_bar_, delta_wx_last_db_*kappa_w_minus_);
      }
    } else { //delta_wx_curr_ != 0.
      if(delta_wx_last_db_==0. || 1e5*delta_wx_last_db_<delta_wx_curr_db_) {
        delta_wx_curr_db_ = kappa_w_plus_bar_ * delta_wx_curr_db_;
      } else {
        delta_wx_curr_db_ = kappa_w_plus_ * delta_wx_curr_db_;
      }
    }
    delta_wx_curr_->setToConstant(delta_wx_curr_db_);

    if(delta_wx_curr_db_ > delta_w_max_bar_) {
      //Hessian perturbation becoming too large
      delta_wx_last_db_ = delta_wd_last_db_ = 0.;
      delta_wx_last_->setToConstant(delta_wx_last_db_);
      delta_wd_last_->setToConstant(delta_wd_last_db_);
      return false;
    }

    delta_wd_curr_db_ = delta_wx_curr_db_;
    delta_wd_curr_->setToConstant(delta_wd_curr_db_);

    delta_wx = delta_wx_curr_db_;
    delta_wd = delta_wd_curr_db_;

    return true;
  }

  double hiopPDPerturbation::compute_delta_c(const double& mu) const
  {
    return delta_c_bar_ * std::pow(mu, kappa_c_);
  }

  /*
  *  class hiopPDPerturbationNormalEqn
  */
  hiopPDPerturbationNormalEqn::hiopPDPerturbationNormalEqn()
    : hiopPDPerturbation(),
      delta_c_min_bar_(1e-20),
      delta_c_max_bar_(1e-2),
      kappa_c_plus_(10.)
  {
  }

  hiopPDPerturbationNormalEqn::~hiopPDPerturbationNormalEqn()
  {
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  bool hiopPDPerturbationNormalEqn::compute_initial_deltas(hiopVector& delta_wx,
                                                           hiopVector& delta_wd,
                                                           hiopVector& delta_cc,
                                                           hiopVector& delta_cd)            
  {
    update_degeneracy_type();
      
    if(delta_wx_curr_db_>0.) {
      delta_wx_last_db_ = delta_wx_curr_db_;      
    }
    if(delta_wd_curr_db_>0.) {
      delta_wd_last_db_ = delta_wd_curr_db_;
    }
    if(delta_cc_curr_db_>0.) {
      delta_cc_last_db_ = delta_cc_curr_db_;      
    }
    if(delta_cd_curr_db_>0.) {
      delta_cd_last_db_= delta_cd_curr_db_;
    }

    if(hess_degenerate_ == dtNotEstablished || jac_degenerate_ == dtNotEstablished) {
      deltas_test_type_ = dttDeltac0Deltaw0;
    } else {
      deltas_test_type_ = dttNoTest;
    }

    delta_cc_curr_db_ = delta_cd_curr_db_ = 0.;
    if(jac_degenerate_ == dtDegenerate) {
      if(!compute_dual_perturb_impl(mu_)) {
        return false;
      }
    }
    delta_cc.setToConstant(delta_cc_curr_db_);
    delta_cd.setToConstant(delta_cd_curr_db_);
    
    delta_wx_curr_db_ = delta_wd_curr_db_ = 0.;
    if(hess_degenerate_ == dtDegenerate) {
      if(!compute_primal_perturb_impl()) {
	      return false;
      }
    }
    delta_wx.setToConstant(delta_wx_curr_db_);
    delta_wd.setToConstant(delta_wd_curr_db_);

    set_delta_curr_vec();
    set_delta_last_vec();

    return true;
  }

  /** Method for correcting inertia */
  bool hiopPDPerturbationNormalEqn::compute_perturb_wrong_inertia(hiopVector& delta_wx,
                                                                  hiopVector& delta_wd,
                                                                  hiopVector& delta_cc,
                                                                  hiopVector& delta_cd)  
  {    
    /** 
    * for normal equation, wrong inertia means the KKT 1x1 matrix is not PD 
    * we try to corret the dual regularization first, and then primal regularizaion
    * */
    update_degeneracy_type();

    assert(delta_wx_curr_db_ == delta_wd_curr_db_);
    assert(delta_cc_curr_db_ == delta_cd_curr_db_);

    bool ret = compute_dual_perturb_impl(mu_);
    if(!ret && delta_wx_curr_db_==0.) {
      delta_cc_curr_db_ = delta_cd_curr_db_ = 0.;
      ret = compute_primal_perturb_impl();
      if(!ret) {
        return ret;
      }
      deltas_test_type_ = dttNoTest;
      if(jac_degenerate_ == dtDegenerate) {
        jac_degenerate_ = dtNotEstablished;
      }
      ret = compute_dual_perturb_impl(mu_);
    }

    delta_wx.setToConstant(delta_wx_curr_db_);
    delta_wd.setToConstant(delta_wd_curr_db_);
    delta_cc.setToConstant(delta_cc_curr_db_);
    delta_cd.setToConstant(delta_cd_curr_db_);

    set_delta_curr_vec();
    return ret;
  }

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  bool hiopPDPerturbationNormalEqn::compute_perturb_singularity(hiopVector& delta_wx,
                                                                hiopVector& delta_wd,
                                                                hiopVector& delta_cc,
                                                                hiopVector& delta_cd)
  {
    /**
     * we try to corret the dual regularization first, and then primal regularizaion
     * same implementation as  hiopPDPerturbationNormalEqn::compute_perturb_wrong_inertia
     */
    return compute_perturb_wrong_inertia(delta_wx, delta_wd, delta_cc, delta_cd);
  }


  /** 
   * Internal method implementing the computation of delta_c
   */
  bool hiopPDPerturbationNormalEqn::compute_dual_perturb_impl(const double& mu)
  {
    assert(delta_cc_curr_db_ == delta_cd_curr_db_ && "these should be equal");
    assert(delta_cc_last_db_ == delta_cd_last_db_ && "these should be equal");
  
    if(delta_cc_curr_db_ == 0.) {
      if(delta_cc_last_db_ == 0.) {
        delta_cc_curr_db_ = std::fmax(delta_c_min_bar_, delta_c_bar_ * std::pow(mu, kappa_c_));
      } else {
        delta_cc_curr_db_ = std::fmax(delta_c_min_bar_, delta_cc_last_db_*kappa_w_minus_);
      }
    } else { //delta_cc_curr_db_ != 0.
      if(delta_cc_last_db_==0. || 1e5*delta_cc_last_db_<delta_cc_curr_db_) {
        delta_cc_curr_db_ = kappa_w_plus_bar_ * delta_cc_curr_db_;
      } else {
        delta_cc_curr_db_ = kappa_c_plus_ * delta_cc_curr_db_;
      }
    }
    delta_cc_curr_->setToConstant(delta_cc_curr_db_);

    if(delta_cc_curr_db_ > delta_w_max_bar_) {
      //dual perturbation becoming too large
      delta_cc_last_db_ = delta_cd_last_db_ = 0.;
      delta_cc_last_->setToConstant(delta_cc_last_db_);
      delta_cd_last_->setToConstant(delta_cd_last_db_);
      return false;
    }
    delta_cd_curr_db_ = delta_cc_curr_db_;
    delta_cd_curr_->setToConstant(delta_cd_curr_db_);

    std::cout << "correct dual (mean): " << delta_cc_curr_db_ << std::endl;

    return true;
  }

  /** 
   * Internal method implementing the computation of delta_w
   */
  bool hiopPDPerturbationNormalEqn::compute_primal_perturb_impl()
  {
    assert(delta_wx_curr_db_ == delta_wd_curr_db_ && "these should be equal");
    assert(delta_wx_last_db_ == delta_wd_last_db_ && "these should be equal");
    if(delta_wx_curr_db_ == 0.) {
      if(delta_wx_last_db_ == 0.) {
        delta_wx_curr_db_ = delta_w_0_bar_;
      } else {
        delta_wx_curr_db_ = std::fmax(delta_w_min_bar_, delta_wx_last_db_*kappa_w_minus_);
      }
    } else { //delta_wx_curr_ != 0.
      if(delta_wx_last_db_==0. || 1e5*delta_wx_last_db_<delta_wx_curr_db_) {
        delta_wx_curr_db_ = kappa_w_plus_bar_ * delta_wx_curr_db_;
      } else {
        delta_wx_curr_db_ = kappa_w_plus_ * delta_wx_curr_db_;
      }
    }
    delta_wx_curr_->setToConstant(delta_wx_curr_db_);

    if(delta_wx_curr_db_ > delta_w_max_bar_) {
      //Hessian perturbation becoming too large
      delta_wx_last_db_ = delta_wd_last_db_ = 0.;
      delta_wx_last_->setToConstant(delta_wx_last_db_);
      delta_wd_last_->setToConstant(delta_wd_last_db_);
      return false;
    }

    delta_wd_curr_db_ = delta_wx_curr_db_;
    delta_wd_curr_->setToConstant(delta_wd_curr_db_);

    std::cout << "correct primal (mean): " << delta_wx_curr_db_ << std::endl;

    return true;
  }

#if 0
    {
       // TODO move set_to_random_constant to PD pertub
    if(nlp_->options->GetString("dual_reg_method") == "unified") {
      dual_reg_->startingAtCopyFromStartingAt(0, delta_cc, 0);
      dual_reg_->startingAtCopyFromStartingAt(neq, delta_cd, 0);
    } else if(nlp_->options->GetString("dual_reg_method") == "randomized") {
      // TODO fix this
      delta_cc
      dual_reg_->set_to_random_constant(0.9*delta_cc, 1.1*delta_cc);
      dual_reg_->startingAtCopyFromStartingAt(0, delta_cc, 0);
      dual_reg_->startingAtCopyFromStartingAt(neq, delta_cd, 0);
    }
#endif
}


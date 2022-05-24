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

namespace hiop
{

  hiopPDPerturbation::~hiopPDPerturbation()
  {}

  /** Initializes and reinitializes object based on the 'options' parameters of the
   * 'nlp_' object.
   * Returns 'false' if something goes wrong, otherwise 'true'
   */
  bool hiopPDPerturbation::initialize(hiopNlpFormulation* nlp)
  {
    delta_w_min_bar_ = nlp->options->GetNumeric("delta_w_min_bar");
    delta_w_max_bar_ = nlp->options->GetNumeric("delta_w_max_bar");
    delta_w_0_bar_   = nlp->options->GetNumeric("delta_0_bar");
    kappa_w_minus_   = nlp->options->GetNumeric("kappa_w_minus");
    kappa_w_plus_bar_= nlp->options->GetNumeric("kappa_w_plus_bar");
    kappa_w_plus_    = nlp->options->GetNumeric("kappa_w_plus");
    delta_c_bar_     = nlp->options->GetNumeric("delta_c_bar");
    kappa_c_         = nlp->options->GetNumeric("kappa_c");

    delta_wx_curr_ = delta_wd_curr_ = 0.;
    delta_cc_curr_ = delta_cd_curr_ = 0.;

    delta_wx_last_ = delta_wd_last_ = 0.;
    delta_cc_last_ = delta_cd_last_;

    num_degen_iters_ = 0;

    deltas_test_type_ = dttNoTest;
    return true;
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  bool hiopPDPerturbation::compute_initial_deltas(double& delta_wx,
                                                  double& delta_wd,
                                                  double& delta_cc,
                                                  double& delta_cd)
  {
    update_degeneracy_type();
      
    if(delta_wx_curr_>0.)
      delta_wx_last_ = delta_wx_curr_;
    if(delta_wd_curr_>0.)
      delta_wd_last_ = delta_wd_curr_;

    if(delta_cc_curr_>0.)
      delta_cc_last_ = delta_cc_curr_;
    if(delta_cd_curr_>0.)
      delta_cd_last_ = delta_cd_curr_;

    if(hess_degenerate_ == dtNotEstablished || jac_degenerate_ == dtNotEstablished) {
      deltas_test_type_ = dttDeltac0Deltaw0;
    } else {
      deltas_test_type_ = dttNoTest;
    }

    if(jac_degenerate_ == dtDegenerate) {
      delta_cc = delta_cd = compute_delta_c(mu_);
    } else {
      delta_cc = delta_cd = 0.;
    }
    delta_cc_curr_ = delta_cc;
    delta_cd_curr_ = delta_cd;

    
    if(hess_degenerate_ == dtDegenerate) {
      delta_wx_curr_ = delta_wd_curr_ = 0.;
      if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
	return false;
      }
    } else {
      delta_wx = delta_wd = 0.;
    }

    delta_wx_curr_ = delta_wx;
    delta_wd_curr_ = delta_wd;
    return true;
  }

  /** Method for correcting inertia */
  bool hiopPDPerturbation::compute_perturb_wrong_inertia(double& delta_wx,
                                                         double& delta_wd,
                                                         double& delta_cc,
                                                         double& delta_cd)
  {    
    update_degeneracy_type();

    assert(delta_wx_curr_ == delta_wd_curr_);
    assert(delta_cc_curr_ == delta_cd_curr_);
    
    delta_cc = delta_cc_curr_;
    delta_cd = delta_cd_curr_;

    bool ret = guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd);
    if(!ret && delta_cc==0.) {
      delta_wx_curr_ = delta_wd_curr_ = 0.;
      delta_cc_curr_ = delta_cd_curr_ = compute_delta_c(mu_);
      deltas_test_type_ = dttNoTest;
      if(hess_degenerate_ == dtDegenerate) {
        hess_degenerate_ = dtNotEstablished;
      }
      
      delta_cc = delta_cc_curr_;
      delta_cd = delta_cd_curr_;
      ret = guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd);
    }
    return ret;
  }

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  bool hiopPDPerturbation::compute_perturb_singularity(double& delta_wx, 
                                                       double& delta_wd,
                                                       double& delta_cc,
                                                       double& delta_cd)
  {    
    assert(delta_wx_curr_ == delta_wd_curr_);
    assert(delta_cc_curr_ == delta_cd_curr_);
    
    if (hess_degenerate_ == dtNotEstablished ||
        jac_degenerate_ == dtNotEstablished) {
      switch (deltas_test_type_) {
      case dttDeltac0Deltaw0:
	//this is the first call
        if (jac_degenerate_ == dtNotEstablished) {
          delta_cc_curr_ = delta_cd_curr_ = compute_delta_c(mu_);
          deltas_test_type_ = dttDeltacposDeltaw0;
        }
        else {
          assert(hess_degenerate_ == dtNotEstablished);
          if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd))
            return false;

          assert(delta_cc == 0. && delta_cd == 0.);
          deltas_test_type_ = dttDeltac0Deltawpos;
        }
        break;
      case dttDeltacposDeltaw0:
        assert(delta_wx_curr_ == 0. && delta_cc_curr_ > 0.);
        assert(jac_degenerate_ == dtNotEstablished);
        delta_cd_curr_ = delta_cc_curr_ = 0.;
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
          return false;
        }
        deltas_test_type_ = dttDeltac0Deltawpos;
        break;
      case dttDeltac0Deltawpos:
        assert(delta_wx_curr_ > 0. && delta_cc_curr_ == 0.);
        delta_cc_curr_ = delta_cd_curr_ = compute_delta_c(mu_);
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
          return false;
        }
        deltas_test_type_ = dttDeltacposDeltawpos;
        break;
      case dttDeltacposDeltawpos:
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd))
          return false;
        break;
      case dttNoTest:
        assert(false && "something went wrong - should not get here");
      }
    } else {
      if(delta_cc_curr_ > 0.) { 
        // If we already used a perturbation for the constraints, we do
        // the same thing as if we were encountering negative curvature
        if(!guts_of_compute_perturb_wrong_inertia(delta_wx, delta_wd)) {
          //todo: need some error message (so that we know what and more important
          //where something went wrong)
          return false;
        }
      } else {
        // Otherwise we now perturb the Jacobian part 
        delta_cd_curr_ = delta_cc_curr_ = compute_delta_c(mu_);
      }
    }

    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;
    delta_cc = delta_cc_curr_;
    delta_cd = delta_cd_curr_;
    
    return true;
  }

  /** Decides degeneracy @hess_degenerate_ and @jac_degenerate_ based on @deltas_test_type_ 
   *  when the @num_degen_iters_ > @num_degen_max_iters_
   */
  void hiopPDPerturbation::update_degeneracy_type()
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

  /** 
   * Internal method implementing the computation of delta_w's to correct wrong inertia
   */
  bool hiopPDPerturbation::guts_of_compute_perturb_wrong_inertia(double& delta_wx, double& delta_wd)
  {
    assert(delta_wx_curr_ == delta_wd_curr_ && "these should be equal");
    assert(delta_wx_last_ == delta_wd_last_ && "these should be equal");
    if(delta_wx_curr_ == 0.) {
      if(delta_wx_last_ == 0.) {
        delta_wx_curr_ = delta_w_0_bar_;
      } else {
        delta_wx_curr_ = std::fmax(delta_w_min_bar_, delta_wx_last_*kappa_w_minus_);
      }
    } else { //delta_wx_curr_ != 0.
      if(delta_wx_last_==0. || 1e5*delta_wx_last_<delta_wx_curr_) {
        delta_wx_curr_ = kappa_w_plus_bar_ * delta_wx_curr_;
      } else {
        delta_wx_curr_ = kappa_w_plus_ * delta_wx_curr_;
      }
    }

    if(delta_wx_curr_ > delta_w_max_bar_) {
      //Hessian perturbation becoming too large
      delta_wx_last_ = delta_wd_last_ = 0.;
      return false;
    }

    delta_wd_curr_ = delta_wx_curr_ ;

    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;

    return true;
  }

  double hiopPDPerturbation::compute_delta_c(const double& mu) const
  {
    return delta_c_bar_ * std::pow(mu, kappa_c_);
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  bool hiopPDPerturbationNormalEqn::compute_initial_deltas(double& delta_wx,
                                                           double& delta_wd,
                                                           double& delta_cc,
                                                           double& delta_cd)            
  {
    update_degeneracy_type();
      
    if(delta_wx_curr_>0.)
      delta_wx_last_ = delta_wx_curr_;
    if(delta_wd_curr_>0.)
      delta_wd_last_ = delta_wd_curr_;

    if(delta_cc_curr_>0.)
      delta_cc_last_ = delta_cc_curr_;
    if(delta_cd_curr_>0.)
      delta_cd_last_ = delta_cd_curr_;

    if(hess_degenerate_ == dtNotEstablished || jac_degenerate_ == dtNotEstablished) {
      deltas_test_type_ = dttDeltac0Deltaw0;
    } else {
      deltas_test_type_ = dttNoTest;
    }

    if(jac_degenerate_ == dtDegenerate) {
      delta_cc_curr_ = delta_cd_curr_ = 0.;
      if(!compute_dual_perturb_impl(mu_, delta_cc, delta_cd)) {
        return false;
      }
    } else {
      delta_cc = delta_cd = 0.;
    }
    delta_cc_curr_ = delta_cc;
    delta_cd_curr_ = delta_cd;


    if(hess_degenerate_ == dtDegenerate) {
      delta_wx_curr_ = delta_wd_curr_ = 0.;
      if(!compute_primal_perturb_impl(mu_, delta_wx, delta_wd)) {
	      return false;
      }
    } else {
      delta_wx = delta_wd = 0.;
    }

    delta_wx_curr_ = delta_wx;
    delta_wd_curr_ = delta_wd;
    return true;
  }

  /** Method for correcting inertia */
  bool hiopPDPerturbationNormalEqn::compute_perturb_wrong_inertia(double& delta_wx,
                                                                  double& delta_wd,
                                                                  double& delta_cc,
                                                                  double& delta_cd)  
  {    
    /** 
    * for normal equation, wrong inertia means the KKT 1x1 matrix is not PD 
    * we try to corret the dual regularization first, and then primal regularizaion
    * */
    update_degeneracy_type();

    assert(delta_wx_curr_ == delta_wd_curr_);
    assert(delta_cc_curr_ == delta_cd_curr_);
    
    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;

    bool ret = compute_dual_perturb_impl(mu_, delta_cc, delta_cd);
    if(!ret && delta_wx==0.) {
      delta_cc_curr_ = delta_cd_curr_ = 0.;
      ret = compute_primal_perturb_impl(delta_wx_curr_, delta_wd_curr_);
      if(!ret) {
        return ret;
      }
      deltas_test_type_ = dttNoTest;
      if(jac_degenerate_ == dtDegenerate) {
        jac_degenerate_ = dtNotEstablished;
      }

      delta_wx = delta_cc_curr_;
      delta_wd = delta_cd_curr_;
      ret = compute_dual_perturb_impl(mu_, delta_cc, delta_cd);
    }
    return ret;
  }

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  bool hiopPDPerturbationNormalEqn::compute_perturb_singularity(double& delta_wx,
                                                                double& delta_wd,
                                                                double& delta_cc,
                                                                double& delta_cd)
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
  bool hiopPDPerturbationNormalEqn::compute_dual_perturb_impl(const double& mu, double& delta_cc, double& delta_cd)
  {
    assert(delta_cc_curr_ == delta_cd_curr_ && "these should be equal");
    assert(delta_cc_last_ == delta_cd_last_ && "these should be equal");
    if(delta_cc_curr_ == 0.) {
      if(delta_cc_last_ == 0.) {
        delta_cc_curr_ = std::fmax(delta_c_min_bar_, delta_c_bar_ * std::pow(mu, kappa_c_));
      } else {
        delta_cc_curr_ = std::fmax(delta_c_min_bar_, delta_cc_last_*kappa_w_minus_);
      }
    } else { //delta_cc_curr_ != 0.
      if(delta_cc_last_==0. || 1e5*delta_cc_last_<delta_cc_curr_) {
        delta_cc_curr_ = kappa_w_plus_bar_ * delta_cc_curr_;
      } else {
        delta_cc_curr_ = kappa_w_plus_ * delta_cc_curr_;
      }
    }

    if(delta_cc_curr_ > delta_w_max_bar_) {
      //dual perturbation becoming too large
      delta_cc_last_ = delta_cd_last_ = 0.;
      return false;
    }

    delta_cd_curr_ = delta_cc_curr_ ;

    delta_cc = delta_cc_curr_;
    delta_cd = delta_cd_curr_;

    return true;
  }

  /** 
   * Internal method implementing the computation of delta_w
   */
  bool hiopPDPerturbationNormalEqn::compute_primal_perturb_impl(double& delta_wx, double& delta_wd)
  {
    assert(delta_wx_curr_ == delta_wd_curr_ && "these should be equal");
    assert(delta_wx_last_ == delta_wd_last_ && "these should be equal");
    if(delta_wx_curr_ == 0.) {
      if(delta_wx_last_ == 0.) {
        delta_wx_curr_ = delta_w_0_bar_;
      } else {
        delta_wx_curr_ = std::fmax(delta_w_min_bar_, delta_wx_last_*kappa_w_minus_);
      }
    } else { //delta_wx_curr_ != 0.
      if(delta_wx_last_==0. || 1e5*delta_wx_last_<delta_wx_curr_) {
        delta_wx_curr_ = kappa_w_plus_bar_ * delta_wx_curr_;
      } else {
        delta_wx_curr_ = kappa_w_plus_ * delta_wx_curr_;
      }
    }

    if(delta_wx_curr_ > delta_w_max_bar_) {
      //Hessian perturbation becoming too large
      delta_wx_last_ = delta_wd_last_ = 0.;
      return false;
    }

    delta_wd_curr_ = delta_wx_curr_ ;

    delta_wx = delta_wx_curr_;
    delta_wd = delta_wd_curr_;

    return true;
  }
}
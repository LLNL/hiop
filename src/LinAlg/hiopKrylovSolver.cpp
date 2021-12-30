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

/* implements the Krylov iterative solver
* @file hiopKrylovSolver.cpp
* @ingroup LinearSolvers
* @author Nai-Yuan Chiang <chiang7@lnnl.gov>, LNNL
* @author Cosmin G. Petra <petra1@lnnl.gov>, LNNL
*/

#include "hiopKrylovSolver.hpp"

#include "hiopVector.hpp"

#include "hiopOptions.hpp"
#include <hiopLinAlgFactory.hpp>
#include "hiopLinearOperator.hpp"

namespace hiop {

  /*
  * class hiopKrylovSolver
  */
  hiopKrylovSolver::hiopKrylovSolver(hiopNlpFormulation* nlp,
                                     hiopLinearOperator* A_opr,
                                     hiopLinearOperator* Mleft_opr,
                                     hiopLinearOperator* Mright_opr,
                                     hiopVector* x0)
    : nlp_{nlp},
      A_opr_{A_opr}, 
      ML_opr_{Mleft_opr},
      MR_opr_{Mright_opr},
      x0_{nullptr}
  {
    if(x0) {
      x0_ = x0->new_copy();
    }
  }
  
  hiopKrylovSolver::~hiopKrylovSolver()
  {
    if(x0_) {
      delete x0_;
    }
  }

  bool hiopKrylovSolver::set_x0(hiopVector& x0)
  {
    if(x0_){
      // just a identity precond
      x0_->copyFrom(x0);
    } else {
      x0_ = x0.new_copy();
    }
    return true;
  }

  /*
  * class hiopPCGSolver
  */
  hiopPCGSolver::hiopPCGSolver(hiopNlpFormulation* nlp,
                               hiopLinearOperator* A_opr,
                               hiopLinearOperator* Mleft_opr,
                               hiopLinearOperator* Mright_opr,
                               hiopVector* x0)
    : hiopKrylovSolver(nlp, A_opr, Mleft_opr, Mright_opr, x0),
      tol_{1e-8},
      maxit_{100},
      iter_{0.0},
      flag_{-1},
      xk_{nullptr},
      xmin_{nullptr},
      res_{nullptr},
      yk_{nullptr},
      zk_{nullptr},
      pk_{nullptr},
      qk_{nullptr}
  {
  }

  hiopPCGSolver::~hiopPCGSolver()
  {
    if(xk_) {
      delete xk_;
      delete xmin_;
      delete res_;
      delete yk_;
      delete zk_;
      delete pk_;
      delete qk_;
    }
  }

bool hiopPCGSolver::solve(hiopVector& b)
{
//  assert(n_ == m_);
//  assert(m_ == b.get_local_size());

  // rhs = 0 --> solution = 0
  double n2b = b.twonorm();
  if(n2b == 0.0) {
    b.setToZero();
    flag_ = 0;
    iter_ = 0.;    
    return true;
  }

  if(xk_==nullptr) {
    xk_ = b.alloc_clone();    //iterate x
    xmin_ = b.alloc_clone();  //iterate which has minimal residual so far
    res_ = b.alloc_clone();   //minimal residual iterate 
    yk_ = b.alloc_clone();    //work vectors
    zk_ = b.alloc_clone();    //work vectors 
    pk_ = b.alloc_clone();    //work vectors
    qk_ = b.alloc_clone();    //work vectors
  }

  //////////////////////////////////////////////////////////////////
  // Starting procedure
  /////////////////////////////////////////////////////////////////
  // starting point
  if(x0_) {
    xk_->copyFrom(*x0_);
  } else {
    xk_->setToZero();
  }

  flag_ = 1;
  index_type imin = 0;        // iteration at which minimal residual is achieved
  double tolb = tol_ * n2b;  // relative tolerance
  
  xmin_->copyFrom(*xk_);
  
  // compute residual: b-KKT*xk
  A_opr_->times_vec(*res_, *xk_);
  res_->axpy(-1.0, b);
  res_->scale(-1.0);               
  double normr = res_->twonorm();  // Norm of residual
  double normr_act = normr;
  double normr_rel;

  // initial guess is good enough
  if(normr <= tolb) { 
    b.copyFrom(*xk_);
    flag_ = 0;
    iter_ = 0.;
    normr_rel = normr / n2b;
    return true;
  }
  
  double normrmin = normr;  // Two-norm of minimum residual
  double rho = 1.0;
  size_type stagsteps = 0;       // stagnation of the method
  size_type moresteps = 0;
  double eps = std::numeric_limits<double>::epsilon();
  
  size_type maxmsteps = 100;//fmin(5, n_-maxit_);
  maxmsteps = 100;//fmin(floor(n_/50), maxmsteps);
  size_type maxstagsteps = 3;


  // main loop for PCG
  double alpha;
  double rho1;
  double pq;
  index_type ii = 0;
  for(; ii < maxit_; ++ii) {
    if(ML_opr_) {
      ML_opr_->times_vec(*yk_, *res_);     
    } else {
      yk_->copyFrom(*res_);
    }
    if(MR_opr_) {
      MR_opr_->times_vec(*zk_, *yk_);      
    } else {
      zk_->copyFrom(*yk_);
    }
    
    rho1 = rho;
    rho = res_->dotProductWith(*zk_);

    //check for stagnation!!!
    if((rho == 0) || abs(rho) > 1E+20) {
      flag_ = 4;
      break;
    }

    if(ii == 0) {
      pk_->copyFrom(*zk_);
    } else {
      double beta = rho / rho1;
      if(beta == 0 || abs(beta) > 1E+20) {
        flag_ = 4;
        break;
      }
      pk_->scale(beta);
      pk_->axpy(1.0, *zk_);      
    }

    A_opr_->times_vec(*qk_, *pk_);
    pq = pk_->dotProductWith(*qk_);
    
    if(pq <= 0.0 || abs(pq) > 1E+20) {
      flag_ = 4;
      break;
    } else {
      alpha = rho / pq;
    }
    if(abs(alpha) > 1E+20) {
      flag_ = 4;
      break;
    }
  
    // Check for stagnation of the method    
    if(pk_->twonorm()*abs(alpha) < eps * xk_->twonorm()) {
      stagsteps++;
    } else {
      stagsteps = 0;
    }

    // new PCG iter
    xk_->axpy(alpha, *pk_);
    res_->axpy(-alpha, *qk_);
    
    normr = res_->twonorm();
    normr_act = normr;

    // check for convergence
    if(normr <= tolb || stagsteps >= maxstagsteps || moresteps) {
      // update residual: b-KKT*xk
      A_opr_->times_vec(*res_,*xk_);
      res_->axpy(-1.0,b);
      res_->scale(-1.0);        
      normr_act = res_->twonorm();

      if(normr_act <= tolb) { 
        b.copyFrom(*xk_);
        flag_ = 0;
        iter_ = ii + 1;
        break;
      } else {
        if(stagsteps >= maxstagsteps && moresteps == 0) {
          stagsteps = 0;
        }
        moresteps++;
        if(moresteps >= maxmsteps) {
          // tol is too small
          flag_ = 3;
          iter_ = ii + 1;
          break;
        }
      }
    }
    // update minimal norm
    if(normr_act < normrmin) {
      normrmin = normr_act;
      xmin_->copyFrom(*xk_);
      imin = ii;
    }
    if(stagsteps >= maxstagsteps) {
      flag_ = 3;
      break;
    }
  } // end of for(; ii < maxit_; ++ii)

  // returned solution is first with minimal residual
  if(flag_ == 0) {
    normr_rel = normr_act/n2b;
    printf("CG converged: actual normResid=%g relResid=%g iter=%.1f\n", 
	         normr_act, normr_rel, iter_);

    b.copyFrom(*xk_);    
  } else {
    // update residual: b-KKT*xk
    A_opr_->times_vec(*res_, *xmin_);
    res_->axpy(-1.0, b);
    res_->scale(-1.0);        
    double normr_comp = res_->twonorm();
    
    if(normr_comp <= normr_act) {
      b.copyFrom(*xmin_);
      iter_ = imin + 1;
      normr_act = normr_comp;
      normr_rel = normr_comp / n2b;
    } else {
      b.copyFrom(*xk_);
      iter_ = ii + 1;
      normr_rel = normr_act / n2b;
    }

    printf("PCG did NOT converged after %.1f iters. The solution from iter %.1f was returned.\n",
           ii+1, imin);
    printf("\t - Error code %d\n\t - Act res=%g\n\t - Rel res=%g \n\n", 
           flag_, normr_act, normr_rel);
  }
  return true; // return true for inertia-free approach
}

} // namespace hiop

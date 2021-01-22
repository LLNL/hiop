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

#include "hiopOptions.hpp"

#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <cstring>

namespace hiop
{

using namespace std;
const char* szDefaultFilename = "hiop.options";

  hiopOptions::hiopOptions(const char* szOptionsFilename/*=NULL*/)
  : log(NULL)
{
  registerOptions();
  loadFromFile(szOptionsFilename==NULL?szDefaultFilename:szOptionsFilename);
  ensureConsistence();
}

hiopOptions::~hiopOptions()
{
  map<std::string, _O*>::iterator it = mOptions.begin();
  for(;it!=mOptions.end(); it++) delete it->second;
}

double hiopOptions::GetNumeric(const char* name) const
{
  map<std::string, _O*>::const_iterator it = mOptions.find(name);
  assert(it!=mOptions.end());
  assert(it->second!=NULL);
  _ONum* option = dynamic_cast<_ONum*>(it->second);
  assert(option!=NULL);
  return option->val;
}

int hiopOptions::GetInteger(const char* name) const
{
  map<std::string, _O*>::const_iterator it = mOptions.find(name);
  assert(it!=mOptions.end());
  assert(it->second!=NULL);
  _OInt* option = dynamic_cast<_OInt*>(it->second);
  assert(option!=NULL);
  return option->val;
}

string hiopOptions::GetString (const char* name) const
{
  map<std::string, _O*>::const_iterator it = mOptions.find(name);
  assert(it!=mOptions.end());
  assert(it->second!=NULL);
  _OStr* option = dynamic_cast<_OStr*>(it->second);
  assert(option!=NULL);
  return option->val;
}

void hiopOptions::registerOptions()
{
  // TODO: add option for mu_target
  registerNumOption("mu0", 1., 1e-16, 1000.,
		    "Initial log-barrier parameter mu (default 1.)");
  registerNumOption("kappa_mu", 0.2, 1e-8, 0.999,
		    "Linear reduction coefficient for mu (default 0.2) (eqn (7) in Filt-IPM paper)");
  registerNumOption("theta_mu", 1.5,  1.0,   2.0,
		    "Exponential reduction coefficient for mu (default 1.5) (eqn (7) in Filt-IPM paper)");
  registerNumOption("eta_phi", 1e-8, 0, 0.01, "Parameter of (suff. decrease) in Armijo Rule");
  registerNumOption("tolerance", 1e-8, 1e-14, 1e-1,
		    "Absolute error tolerance for the NLP (default 1e-8)");
  registerNumOption("rel_tolerance", 0., 0., 0.1,
		    "Error tolerance for the NLP relative to errors at the initial point. A null "
		    "value disables this option (default 0.)");
  registerNumOption("tau_min", 0.99, 0.9,  0.99999,
		    "Fraction-to-the-boundary parameter used in the line-search to back-off a bit "
		    "(see eqn (8) in the Filt-IPM paper) (default 0.99)");
  registerNumOption("kappa_eps", 10., 1e-6, 1e+3,
		    "mu is reduced when when log-bar error is below kappa_eps*mu (default 10.)");
  registerNumOption("kappa1", 1e-2, 1e-16, 1e+0,
		    "sufficiently-away-from-the-boundary projection parameter used in initialization (default 1e-2)");
  registerNumOption("kappa2", 1e-2, 1e-16, 0.49999,
		    "shift projection parameter used in initialization for double-bounded variables (default 1e-2)");
  registerNumOption("smax", 100., 1., 1e+7,
		    "multiplier threshold used in computing the scaling factors for the optimality error (default 100.)");

  {
    // 'dualsUpdateType' should be 'lsq' or 'linear' for  'Hessian=quasinewton_approx'
    // 'dualsUpdateType' can only be 'linear' for Newton methods 'Hessian=analytical_exact'

    //here will set the default value to 'lsq' and this will be adjusted later in 'ensureConsistency'
    //to a valid value depending on the 'Hessian' value
    vector<string> range(2); range[0]="lsq"; range[1]="linear";
    registerStrOption("dualsUpdateType", "lsq", range,
		      "Type of update of the multipliers of the eq. constraints "
                      "(default is 'lsq' when 'Hessian' is 'quasinewton_approx' and "
                      "'linear' when 'Hessian is 'analytical_exact')"); 
  }
  {
    vector<string> range(2); range[0]="lsq"; range[1]="zero";
    registerStrOption("dualsInitialization", "lsq", range,
		      "Type of initialization of the multipliers of the eq. cons. (default lsq)");
  }

  registerIntOption("max_iter", 3000, 1, 1e6, "Max number of iterations (default 3000)");

  registerNumOption("acceptable_tolerance", 1e-6, 1e-14, 1e-1,
		    "HiOp will terminate if the NLP residuals are below for 'acceptable_iterations' "
		    "many consecutive iterations (default 1e-6)");
  registerIntOption("acceptable_iterations", 10, 1, 1e6,
		    "Number of iterations of acceptable tolerance after which HiOp terminates (default 10)");

  registerNumOption("sigma0", 1., 0., 1e+7,
		    "Initial value of the initial multiplier of the identity in the secant "
		    "approximation (default 1.)");
  {
    vector<string> range(2); range[0] = "no"; range[1] = "yes";
    registerStrOption("accept_every_trial_step", "no", range,
		      "Disable line-search and take close-to-boundary step");
  }
  {
    vector<string> range(5);
    range[0]="sigma0"; range[1]="sty"; range[2]="sty_inv";
    range[3]="snrm_ynrm";  range[4]="sty_srnm_ynrm";
    registerStrOption("sigma_update_strategy", range[1], range,
		      "Updating strategy for the multiplier of the identity in the secant "
		      "approximation (default sty)");
  }
  registerIntOption("secant_memory_len", 6, 0, 256,
		    "Size of the memory of the Hessian secant approximation");

  registerIntOption("verbosity_level", 3, 0, 12,
		    "Verbosity level: 0 no output (only errors), 1=0+warnings, 2=1 (reserved), "
		    "3=2+optimization output, 4=3+scalars; larger values explained in hiopLogger.hpp");

  {
    vector<string> range(3); range[0]="remove"; range[1]="relax"; range[2]="none";
    registerStrOption("fixed_var", "none", range,
		      "Treatment of fixed variables: 'remove' from the problem, 'relax' bounds "
		      "by 'fixed_var_perturb', or 'none', in which case the HiOp will terminate "
		      "with an error message if fixed variables are detected (default 'none'). "
                      "Value 'remove' is available only when 'compute_mode' is 'hybrid' or 'cpu'.");

    registerNumOption("fixed_var_tolerance", 1e-15, 1e-30, 0.01,
		      "A variable is considered fixed if |upp_bnd-low_bnd| < fixed_var_tolerance * "
		      "max(abs(upp_bnd),1) (default 1e-15)");

    registerNumOption("fixed_var_perturb", 1e-8, 1e-14, 0.1,
		      "Perturbation of the lower and upper bounds for fixed variables relative "
		      "to its magnitude: lower/upper_bound -=/+= max(abs(upper_bound),1)*"
		      "fixed_var_perturb (default 1e-8)");
  }

  //optimization method used
  {
    vector<string> range(2); range[0]="quasinewton_approx"; range[1]="analytical_exact";
    registerStrOption("Hessian", "quasinewton_approx", range,
		      "Type of Hessian used with the filter IPM: 'quasinewton_approx' built internally "
		      "by HiOp (default option) or 'analytical_exact' provided by the user");
  }
  //linear algebra
  {
    vector<string> range(4); range[0] = "auto"; range[1]="xycyd"; range[2]="xdycyd"; range[3]="full";
    registerStrOption("KKTLinsys", "auto", range,
		      "Type of KKT linear system used internally: decided by HiOp 'auto' "
		      "(default option), the more compact 'XYcYd, the more stable 'XDYcYd', or the "
                      "full-size non-symmetric 'full'. The last three options are only available with "
                      "'Hessian=analyticalExact'.");
  }
  {
    vector<string> range(3); range[0]="stable"; range[1]="speculative"; range[2]="forcequick";
    registerStrOption("linsol_mode", "stable", range,
                      "'stable'=using stable factorization; "
                      "'speculative'=try faster linear solvers when is detected to be safe "
                      "to do so (experimental) ; "
                      "'forcequick'=rely on faster solvers on all situations "
                      "(experimental, avoid)");
  }

  //factorization acceptor
  {
    vector<string> range(2); range[0] = "inertia_correction"; range[1]="inertia_free";
    registerStrOption("fact_acceptor", "inertia_correction", range,
                      "The criteria used to accept a factorization: "
                      " inertia_correction (default option) --- check if inertia is correct. "
                      " inertia_free --- to be updated");
  }  
  //computations
  {
    vector<string> range(4); range[0]="auto"; range[1]="cpu"; range[2]="hybrid"; range[3]="gpu";
    registerStrOption("compute_mode", "auto", range,
		      "'auto', 'cpu', 'hybrid', 'gpu'; 'hybrid'=linear solver on gpu; 'auto' will decide between "
		      "'cpu', 'gpu' and 'hybrid' based on the other options passed");
  }
  //inertia correction and Jacobian regularization
  {
    //Hessian related
    registerNumOption("delta_w_min_bar", 1e-20, 0, 1000.,
		      "Smallest perturbation of the Hessian block for inertia correction");
    registerNumOption("delta_w_max_bar", 1e+20, 1e-40, 1e+40,
		      "Largest perturbation of the Hessian block for inertia correction");
    registerNumOption("delta_0_bar", 1e-4, 0, 1e+40,
		      "First perturbation of the Hessian block for inertia correction");
    registerNumOption("kappa_w_minus", 1./3, 1e-20, 1-1e-20,
		      "Factor to decrease the most recent successful perturbation for inertia correction");
    registerNumOption("kappa_w_plus", 8., 1+1e-20, 1e+40,
		      "Factor to increase perturbation when it did not provide correct "
		      "inertia correction (not first iteration)");
    registerNumOption("kappa_w_plus_bar", 100., 1+1e-20, 1e+40,
		      "Factor to increase perturbation when it did not provide correct "
		      "inertia correction (first iteration when scale not known)");
    //Jacobian related
    registerNumOption("delta_c_bar", 1e-8, 1e-20, 1e+40,
		      "Factor for regularization for potentially rank-deficient Jacobian "
		      "(delta_c=delta_c_bar*mu^kappa_c");
    registerNumOption("kappa_c", 0.25, 0., 1e+40,
		      "Exponent of mu when computing regularization for potentially rank-deficient "
		      "Jacobian (delta_c=delta_c_bar*mu^kappa_c)");

  }
  // perfromance profiling
  {
    vector<string> range(2);
    range[0] = "on";
    range[1] = "off";
    registerStrOption("time_kkt", "off", range,
		      "turn on/off performance timers and reporting of the computational constituents of the "
		      "KKT solve process");
  }

  //other options
  {
    vector<string> range(2); range[0]="no"; range[1]="yes";
    registerStrOption("write_kkt", range[0], range,
		      "write internal KKT linear system (matrix, rhs, sol) to file (default 'no')");
  }

  // memory space selection
  {
#ifdef HIOP_USE_RAJA
    vector<string> range(4);
    range[0] = "default";
    range[1] = "host";
    range[2] = "device";
    range[3] = "um";
#else
    vector<string> range(1);
    range[0] = "default";
#endif
    registerStrOption("mem_space", range[0], range,
    "Determines the memory space in which future linear algebra objects will be created");
  }
}

void hiopOptions::registerNumOption(const std::string& name, double defaultValue,
				    double low, double upp, const char* description)
{
  mOptions[name]=new _ONum(defaultValue, low, upp, description);
}

void hiopOptions::registerStrOption(const std::string& name, const std::string& defaultValue,
				    const std::vector<std::string>& range, const char* description)
{
  mOptions[name]=new _OStr(defaultValue, range, description);
}

void hiopOptions::registerIntOption(const std::string& name,
				    int defaultValue,
				    int low,
				    int upp,
				    const char* description)
{
  mOptions[name]=new _OInt(defaultValue, low, upp, description);
}

void hiopOptions::ensureConsistence()
{
  //check that the values of different options are consistent
  //do not check is the values of a particular option is valid; this is done in the Set methods
  double eps_tol_accep = GetNumeric("acceptable_tolerance");
  double eps_tol  =      GetNumeric("tolerance");
  if(eps_tol_accep < eps_tol) {
    if(is_user_defined("acceptable_tolerance")) {
      log_printf(hovWarning,
                 "There is no reason to set 'acceptable_tolerance' tighter than 'tolerance'. "
                 "Will set the two to 'tolerance'.\n");
      set_val("acceptable_tolerance", eps_tol);
    }
  }

  if(GetString("Hessian")=="quasinewton_approx") {
    string strKKT = GetString("KKTLinsys");
    if(strKKT=="xycyd" || strKKT=="xdycyd" || strKKT=="full") {
      if(is_user_defined("Hessian")) {
        log_printf(hovWarning,
                   "The option 'KKTLinsys=%s' is not valid with 'Hessian=quasiNewtonApprox'. "
                   "Will use 'KKTLinsys=auto'\n", strKKT.c_str());
      }
      set_val("KKTLinsys", "auto");
    }
  }

  if(GetString("Hessian")=="analytical_exact") {
    string duals_update_type = GetString("dualsUpdateType");
    if("linear" != duals_update_type) {
      // 'dualsUpdateType' should be 'lsq' or 'linear' for  'Hessian=quasinewton_approx'
      // 'dualsUpdateType' can only be 'linear' for Newton methods 'Hessian=analytical_exact'

      //warn only if these are defined by the user (option file or via SetXXX methods)
      if(is_user_defined("dualsUpdateType")) {
        log_printf(hovWarning,
                   "The option 'dualsUpdateType=%s' is not valid with 'Hessian=analytical_exact'. "
                   "Will use 'dualsUpdateType=linear'.\n",
                   duals_update_type.c_str());
      }
      set_val("dualsUpdateType", "linear");
    }
  }

// When RAJA is not enabled ...
#ifndef HIOP_USE_RAJA
  if(GetString("compute_mode")=="gpu") {
    if(is_user_defined("compute_mode")) {
      log_printf(hovWarning,
                 "option compute_mode=gpu was changed to 'hybrid' since HiOp was built without "
                 "RAJA/Umpire support.\n");
    }
    set_val("compute_mode", "hybrid");
  }
  if(GetString("mem_space")!="default") {
    std::string memory_space = GetString("mem_space");
    if(is_user_defined("compute_mode")) {
      log_printf(hovWarning,
                 "option mem_space=%s was changed to 'default' since HiOp was built without "
                 "RAJA/Umpire support.\n", memory_space.c_str());
    }
    set_val("mem_space", "default");
  }
#endif

  // No removing of fixed variables in GPU compute mode ...
  if(GetString("compute_mode")=="gpu") {
    if(GetString("fixed_var")=="remove") {
      
      log_printf(hovWarning,
                 "option fixed_var=remove was changed to 'relax' since only 'relax'"
                 "is supported in GPU compute mode.\n");
      set_val("fixed_var", "relax");
    }
  }
  
// No hybrid or GPU compute mode if HiOp is built without GPU linear solvers
#ifndef HIOP_USE_MAGMA
#ifndef HIOP_USE_STRUMPACK
  if(GetString("compute_mode")=="hybrid") {

    if(is_user_defined("compute_mode")) {
      log_printf(hovWarning,
                 "option compute_mode=hybrid was changed to 'cpu' since HiOp was built without "
                 "GPU support/Magma.\n");
    }
    set_val("compute_mode", "cpu");
  }
  if(GetString("compute_mode")=="gpu") {
    log_printf(hovWarning,
	       "option compute_mode=gpu was changed to 'cpu' since HiOp was built without "
	       "GPU support/Magma.\n");
    set_val("compute_mode", "cpu");
  }
#endif
#endif
}

static inline std::string &ltrim(std::string &s) {
  //s.erase(s.begin(), std::find_if(s.begin(), s.end(),
  //          std::not1(std::ptr_fun<int, int>(std::isspace))));
  s.erase(s.begin(),
	  std::find_if(s.begin(),
		       s.end(),
		       [](int c) {return !std::isspace(c);}
		       )
	  );
    return s;
}

void hiopOptions::loadFromFile(const char* filename)
{
  if(NULL==filename) {
    log_printf(hovError, "Option file name not valid");
    return;
  }

  ifstream input( filename );

  if(input.fail())
    if(strcmp(szDefaultFilename, filename)) {
      log_printf(hovError,
		 "Failed to read option file '%s'. Hiop will use default options.\n",
		 filename);
      return;
    }

  string line; string name, value;
  for( std::string line; getline( input, line ); ) {

    line = ltrim(line);

    if(line.size()==0) continue;
    if(line[0]=='#') continue;

    istringstream iss(line);
    if(!(iss >> name >> value)) {
      log_printf(hovWarning,
		 "Hiop could not parse and ignored line '%s' from the option file\n",
		 line.c_str());
      continue;
    }

    //find the _O object in mOptions corresponding to 'optname' and set his value to 'optval'
    _ONum* on; _OInt* oi; _OStr* os;

    map<string, _O*>::iterator it = mOptions.find(name);
    if(it!=mOptions.end()) {
      _O* option = it->second;
      on = dynamic_cast<_ONum*>(option);
      if(on!=NULL) {
	stringstream ss(value); double val;
	if(ss>>val) { SetNumericValue(name.c_str(), val, true); }
	else
	  log_printf(hovWarning,
		      "Hiop could not parse value '%s' as double for option '%s' specified in "
		     "the option file and will use default value '%g'\n",
		      value.c_str(), name.c_str(), on->val);
      } else {
	os = dynamic_cast<_OStr*>(option);
	if(os!=NULL) {
	  SetStringValue(name.c_str(), value.c_str(), true);
	} else {
	  oi = dynamic_cast<_OInt*>(option);
	  if(oi!=NULL) {
	    stringstream ss(value); int val;
	    if(ss>>val) { SetIntegerValue(name.c_str(), val, true); }
	    else {
	      log_printf(hovWarning,
			  "Hiop could not parse value '%s' as int for option '%s' specified in "
			 "the option file and will use default value '%d'\n",
			  value.c_str(), name.c_str(), oi->val);
	    }
	  } else {
	    // not one of the expected types? Can't happen
	    assert(false);
	  }
	}
      }

    } else { // else from it!=mOptions.end()
      // option not recognized/found/registered
      log_printf(hovWarning,
		 "Hiop does not understand option '%s' specified in the option file and will "
		 "ignore its value '%s'.\n", name.c_str(), value.c_str());
    }
  } //end of the for over the lines
}

bool hiopOptions::is_user_defined(const char* option_name)
{
  map<string, _O*>::iterator it = mOptions.find(option_name);
  if(it==mOptions.end()) {
    return false;
  }
  return (it->second->specifiedInFile || it->second->specifiedAtRuntime);
}

bool hiopOptions::set_val(const char* name, const double& value)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _ONum* option = dynamic_cast<_ONum*>(it->second);
    if(NULL==option) {
      assert(false && "mismatch between name and type happened in internal 'set_val'");
    } else {

      if(value<option->lb || value>option->ub) {
        assert(false && "incorrect use of internal 'set_val': value out of bounds\n");
      } else {
        option->val = value;
      }
    }
  } else {
    assert(false && "trying to change an inexistent option with internal 'set_val'");
  }
  return true;
}
bool hiopOptions::SetNumericValue (const char* name, const double& value, const bool& setFromFile/*=false*/)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _ONum* option = dynamic_cast<_ONum*>(it->second);
    if(NULL==option) {
      log_printf(hovWarning,
		"Hiop does not know option '%s' as 'numeric'. Maybe it is an 'integer' or 'string' "
		 "value? The option will be ignored.\n", name);
    } else {
      if(true==option->specifiedInFile) {
	if(false==setFromFile) {
	  log_printf(hovWarning,
		     "Hiop will ignore value '%g' set for option '%s' at runtime since this option is "
		     "already specified in the option file.\n", value, name);
	  return true;
	}
      }

      if(setFromFile) {
	option->specifiedInFile=true;
      } else {
        option->specifiedAtRuntime=true;
      }

      if(value<option->lb || value>option->ub) {
	log_printf(hovWarning,
		   "Hiop: option '%s' must be in [%g,%g]. Default value %g will be used.\n",
		   name, option->lb, option->ub, option->val);
      } else option->val = value;
    }
  } else {
    log_printf(hovWarning,
		"Hiop does not understand option '%s' and will ignore its value '%g'.\n",
		name, value);
  }
  ensureConsistence();
  return true;
}


bool hiopOptions::set_val(const char* name, const int& value)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OInt* option = dynamic_cast<_OInt*>(it->second);
    if(NULL==option) {
      assert(false && "mismatch between name and type happened in internal 'set_val'");
    } else {

      if(value<option->lb || value>option->ub) {
        assert(false && "incorrect use of internal 'set_val': value out of bounds\n");
      } else {
        option->val = value;
      }
    }
  } else {
    assert(false && "trying to change an inexistent option with internal 'set_val'");
  }
  return true;
}


bool hiopOptions::SetIntegerValue(const char* name, const int& value, const bool& setFromFile/*=false*/)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OInt* option = dynamic_cast<_OInt*>(it->second);
    if(NULL==option) {
      log_printf(hovWarning,
		 "Hiop does not know option '%s' as 'integer'. Maybe it is an 'numeric' "
		 "or a 'string' option? The option will be ignored.\n",
		  name);
    } else {
      if(true==option->specifiedInFile) {
	if(false==setFromFile) {
	  log_printf(hovWarning,
		     "Hiop will ignore value '%d' set for option '%s' at runtime since this "
		     "option is already specified in the option file.\n", value, name);
	  return true;
	}
      }

      if(setFromFile)
	option->specifiedInFile=true;

      if(value<option->lb || value>option->ub) {
	log_printf(hovWarning,
		   "Hiop: option '%s' must be in [%d, %d]. Default value %d will be used.\n",
		   name, option->lb, option->ub, option->val);
      } else option->val = value;
    }
  } else {
    log_printf(hovWarning,
		"Hiop does not understand option '%s' and will ignore its value '%d'.\n",
		name, value);
  }
  ensureConsistence();
  return true;
}

bool hiopOptions::set_val(const char* name, const char* value_in)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OStr* option = dynamic_cast<_OStr*>(it->second);
    if(NULL==option) {
      assert(false && "mismatch between name and type happened in internal 'set_val'");
    } else {
      string value(value_in);
      transform(value.begin(), value.end(), value.begin(), ::tolower);
      //see if it is in the range (of supported values)
      bool inrange=false;
      for(int it=0; it<option->range.size() && !inrange; it++) inrange = (option->range[it]==value);

      if(!inrange) {
        assert(false && "incorrect use of internal 'set_val': value out of range\n");
      } else {
        option->val = value;
      }
    }
  } else {
    assert(false && "trying to change an inexistent option with internal 'set_val'");
  }
  return true;
}

bool hiopOptions::SetStringValue (const char* name,  const char* value, const bool& setFromFile/*=false*/)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OStr* option = dynamic_cast<_OStr*>(it->second);
    if(NULL==option) {
      log_printf(hovWarning,
		  "Hiop does not know option '%s' as 'string'. Maybe it is an 'integer' or a "
		 "'string' option? The option will be ignored.\n", name);
    } else {
      if(true==option->specifiedInFile) {
	if(false==setFromFile) {
	  log_printf(hovWarning,
		     "Hiop will ignore value '%s' set for option '%s' at runtime since this option "
		     "is already specified in the option file.\n", value, name);
	  return true;
	}
      }

      if(setFromFile)
	option->specifiedInFile=true;

      string strValue(value);
      transform(strValue.begin(), strValue.end(), strValue.begin(), ::tolower);
      //see if it is in the range (of supported values)
      bool inrange=false;
      for(int it=0; it<option->range.size() && !inrange; it++) inrange = (option->range[it]==strValue);

      if(!inrange) {
	stringstream ssRange; ssRange << " ";
	for(int it=0; it<option->range.size(); it++) ssRange << option->range[it] << " ";

	log_printf(hovWarning,
		    "Hiop: value '%s' for option '%s' must be one of [%s]. Default value "
		   "'%s' will be used.\n", value, name, ssRange.str().c_str(), option->val.c_str());
      } else option->val = strValue;
    }
  } else {
    log_printf(hovWarning,
		"Hiop does not understand option '%s' and will ignore its value '%s'.\n",
		name, value);
  }
  ensureConsistence();
  return true;
}

void hiopOptions::log_printf(hiopOutVerbosity v, const char* format, ...)
{
  char buff[1024];
  va_list args;
  va_start (args, format);
  vsprintf (buff,format, args);
  if(log)
    log->printf(v,buff);
  else
    hiopLogger::printf_error(v,buff);
  //fprintf(stderr,buff);
  va_end (args);
}

void hiopOptions::print(FILE* file, const char* msg) const
{
  if(NULL==msg) fprintf(file, "#\n# Hiop options\n#\n");
  else          fprintf(file, "%s ", msg);

  map<string,_O*>::const_iterator it = mOptions.begin();
  for(; it!=mOptions.end(); it++) {
    fprintf(file, "%s ", it->first.c_str());
    it->second->print(file);
    fprintf(file, "\n");
  }
  fprintf(file, "# end of Hiop options\n\n");
}

void hiopOptions::_ONum::print(FILE* f) const
{
  fprintf(f, "%.3e \t# (numeric) %g to %g [%s]", val, lb, ub, descr.c_str());
}
void hiopOptions::_OInt::print(FILE* f) const
{
  fprintf(f, "%d \t# (integer)  %d to %d [%s]", val, lb, ub, descr.c_str());
}

void hiopOptions::_OStr::print(FILE* f) const
{
  stringstream ssRange; ssRange << " ";
  for(int i=0; i<range.size(); i++) ssRange << range[i] << " ";
  fprintf(f, "%s \t# (string) one of [%s] [%s]", val.c_str(), ssRange.str().c_str(), descr.c_str());
}



} //~end namespace

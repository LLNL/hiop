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

hiopOptions::hiopOptions(hiopLogger* log_, const char* szOptionsFilename/*=NULL*/)
  : log(log_)
{
  registerOptions();
  loadFromFile(szOptionsFilename==NULL?szDefaultFilename:szOptionsFilename);
  ensureConsistence();
  log->write(NULL, *this, hovSummary);
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
  registerNumOption("mu0", 1., 1e-6, 1000., "Initial log-barrier parameter mu (default 1.)");
  registerNumOption("kappa_mu", 0.2, 1e-8, 0.999, "Linear reduction coefficient for mu (default 0.2) (eqn (7) in Filt-IPM paper)");
  registerNumOption("theta_mu", 1.5,  1.0,   2.0, "Exponential reduction coefficient for mu (default 1.5) (eqn (7) in Filt-IPM paper)");
  registerNumOption("tolerance", 1e-6, 1e-14, 1e-1, "Absolute error tolerance for the NLP (default 1e-6)");
  registerNumOption("tau_min", 0.99, 0.9,  0.99999, "Fraction-to-the-boundary parameter used in the line-search to back-off a bit (default 0.99) (eqn (8) in the Filt-IPM paper");
  registerNumOption("kappa_eps", 10., 1e-6, 1e+3, "mu is reduced when when log-bar error is below kappa_eps*mu (default 10.)");
  registerNumOption("kappa1", 1e-2, 1e-8, 1e+0, "sufficiently-away-from-the-boundary projection parameter used in initialization (default 1e-2)");
  registerNumOption("kappa2", 1e-2, 1e-8, 0.49999, "shift projection parameter used in initialization for double-bounded variables (default 1e-2)");
  registerNumOption("smax", 100., 1., 1e+7, "multiplier threshold used in computing the scaling factors for the optimality error (default 100.)"); 

  {
    vector<string> range(2); range[0]="lsq"; range[1]="linear";
    registerStrOption("dualsUpdateType", "lsq", range, "Type of update of the multipliers of the eq. cons. (default lsq)"); //
  }
  {
    vector<string> range(2); range[0]="lsq"; range[1]="zero";
    registerStrOption("dualsInitialization", "lsq", range, "Type of update of the multipliers of the eq. cons. (default lsq)");
  }

  registerIntOption("max_iter", 1000, 1, 1e6, "Max number of iterations (default 1000)");
}

void hiopOptions::registerNumOption(const std::string& name, double defaultValue, double low, double upp, const char* description)
{
  mOptions[name]=new _ONum(defaultValue, low, upp, description);
}

void hiopOptions::registerStrOption(const std::string& name, const std::string& defaultValue, const std::vector<std::string>& range, const char* description)
{
  mOptions[name]=new _OStr(defaultValue, range, description);
}

void hiopOptions::registerIntOption(const std::string& name, int    defaultValue, int low, int upp, const char* description)
{
  mOptions[name]=new _OInt(defaultValue, low, upp, description);
}

void hiopOptions::ensureConsistence()
{
  //check that the values of different options are consistent 
  //do not check is the values of a particular option is valid; this is done in the Set methods
}

static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

void hiopOptions::loadFromFile(const char* filename)
{
  if(NULL==filename) { log->printf(hovError, "Option file name not valid"); return;}

  ifstream input( filename );

  if(input.fail()) 
    if(strcmp(szDefaultFilename, filename)) {
      log->printf(hovError, "Failed to read option file '%s'. Hiop will use default options.\n", filename);
      return;
    }

  string line; string name, value;
  for( std::string line; getline( input, line ); ) {

    line = ltrim(line);

    if(line.size()==0) continue;
    if(line[0]=='#') continue;

    istringstream iss(line);
    if(!(iss >> name >> value)) {
      log->printf(hovWarning, "Hiop could not parse and ignored line '%s' from the option file\n", line.c_str());
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
	if(ss>>val) { SetNumericValue(name.c_str(), val); }
	else 
	  log->printf(hovWarning, 
		      "Hiop could not parse value '%s' as double for option '%s' specified in the option file and will use default value '%g'\n", 
		      value.c_str(), name.c_str(), on->val);
      } else {
	os = dynamic_cast<_OStr*>(option);
	if(os!=NULL) {
	  SetStringValue(name.c_str(), value.c_str());
	} else {
	  oi = dynamic_cast<_OInt*>(option);
	  if(oi!=NULL) {
	    stringstream ss(value); int val;
	    if(ss>>val) { SetIntegerValue(name.c_str(), val); }
	    else {
	      log->printf(hovWarning, 
			  "Hiop could not parse value '%s' as int for option '%s' specified in the option file and will use default value '%d'\n",
			  value.c_str(), name.c_str(), oi->val);
	    }
	  } else {
	    // not any of the types? Can't happen
	    assert(false);
	  }
	}
      }

    } else { // else from it!=mOptions.end()
      // option not recognized/found/registered
      log->printf(hovWarning, 
		  "Hiop does not understand option '%s' specified in the option file and will ignore its value '%s'.\n",
		  name.c_str(), value.c_str());
    }
  } //end of the for over the lines
}

void hiopOptions::SetNumericValue (const char* name, const double& value)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _ONum* option = dynamic_cast<_ONum*>(it->second);
    if(NULL==option) {
      log->printf(hovWarning, 
		"Hiop does not know option '%s' as 'numeric'. Maybe it is an 'integer' value? .\n",
		name);
    } else {
      if(value<option->lb || value>option->ub)
	log->printf(hovWarning, 
		    "Hiop: option '%s' must be in [%g,%g]. Default value %g will be used.\n",
		    name, option->lb, option->ub, option->val);
      else option->val = value;
    }
  } else {
    log->printf(hovWarning, 
		"Hiop does not understand option '%s' and will ignore its value '%g'.\n",
		name, value);
  }
}

void hiopOptions::SetIntegerValue(const char* name, const int& value)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OInt* option = dynamic_cast<_OInt*>(it->second);
    if(NULL==option) {
      log->printf(hovWarning, 
		  "Hiop does not know option '%s' as 'integer'. Maybe it is an 'numeric' or a 'string' option? .\n",
		  name);
    } else {
      if(value<option->lb || value>option->ub)
	log->printf(hovWarning, 
		    "Hiop: option '%s' must be in [%d, %d]. Default value %d will be used.\n",
		    name, option->lb, option->ub, option->val);
      else option->val = value;
    }
  } else {
    log->printf(hovWarning, 
		"Hiop does not understand option '%s' and will ignore its value '%g'.\n",
		name, value);
  }
}

void hiopOptions::SetStringValue (const char* name,  const char* value)
{
  map<string, _O*>::iterator it = mOptions.find(name);
  if(it!=mOptions.end()) {
    _OStr* option = dynamic_cast<_OStr*>(it->second);
    if(NULL==option) {
      log->printf(hovWarning, 
		  "Hiop does not know option '%s' as 'string'. Maybe it is an 'integer' or a 'string' option? .\n",
		  name);
    } else {
      string strValue(value);
      transform(strValue.begin(), strValue.end(), strValue.begin(), ::tolower);
      //see if it is in the range (of supported values)
      bool inrange=false;
      for(int it=0; it<option->range.size() && !inrange; it++) inrange = (option->range[it]==strValue);

      if(!inrange) {
	stringstream ssRange; ssRange << " ";
	for(int it=0; it<option->range.size(); it++) ssRange << option->range[it] << " ";

	log->printf(hovWarning, 
		    "Hiop: value '%s' for option '%s' must be one of [%s]. Default value '%s' will be used.\n",
		    value, name, ssRange.str().c_str(), option->val.c_str());
      }
      else option->val = value;
    }
  } else {
    log->printf(hovWarning, 
		"Hiop does not understand option '%s' and will ignore its value '%g'.\n",
		name, value);
  }
}


void hiopOptions::print(FILE* file, const char* msg) const
{
  if(NULL==msg) fprintf(file, "#\n# Hiop options\n#\n");
  else          fprintf(file, "%s ", msg);
 
  map<string,_O*>::const_iterator it = mOptions.begin();
  for(; it!=mOptions.end(); it++) {
    fprintf(file, "%s  ", it->first.c_str());
    it->second->print(file);
    fprintf(file, "\n");
  }
  fprintf(file, "# end of Hiop options\n\n");
}

void hiopOptions::_ONum::print(FILE* f) const
{
  fprintf(f, "%e     \t # (numeric)  %g to %g   [%s]", val, lb, ub, descr.c_str());
}
void hiopOptions::_OInt::print(FILE* f) const
{
  fprintf(f, "%d     \t # (integer)  %d to %d   [%s]", val, lb, ub, descr.c_str());
}

void hiopOptions::_OStr::print(FILE* f) const
{
  stringstream ssRange; ssRange << " ";
  for(int i=0; i<range.size(); i++) ssRange << range[i] << " ";
  fprintf(f, "%s     \t # (string) one of [%s]   [%s]", val.c_str(), ssRange.str().c_str(), descr.c_str());
}



} //~end namespace

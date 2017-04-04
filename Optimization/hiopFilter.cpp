#include "hiopFilter.hpp"

using namespace std;

namespace hiop
{

bool hiopFilter::contains(const double& theta, const double& phi) const
{
  list<FilterEntry>::const_iterator it = entries.begin();
  bool bFound=false;
  while(it!=entries.end()) {
    if(theta>=it->theta && phi>=it->phi) { 
      bFound=true;
      break;
    }
    ++it;
  }
  return bFound;
}

};

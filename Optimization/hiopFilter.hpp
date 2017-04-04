#ifndef HIOP_FILTER
#define HIOP_FILTER

#include <list>
#include <cassert>

namespace hiop
{

class hiopFilter
{
public:
  hiopFilter()  { };
  ~hiopFilter() { };
  inline void initialize  (const double& theta_max) { entries.clear(); entries.push_front(FilterEntry(theta_max,-1e20)); }
  inline void reinitialize(const double& theta_max) { initialize(theta_max); }
  //entries are pushed at the front since these are most likely to reject new iterates
  inline void add(const double& theta, const double& phi) { entries.push_front(FilterEntry(theta,phi)); }
  bool contains(const double& theta, const double& phi) const;

private:
  struct FilterEntry { 
    FilterEntry(const double& t, const double& p) : theta(t), phi(p) {};
    double theta,phi; 
#ifdef DEEP_CHECKING
    FilterEntry() : theta(0.), phi(0.) { assert(true); }
#endif
  };
  std::list<FilterEntry> entries;
};

};
#endif

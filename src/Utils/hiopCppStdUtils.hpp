#ifndef HIOP_STD_UTILS
#define HIOP_STD_UTILS

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <vector>
#include <list>
#include <numeric>
#include <cassert>
#include <math.h>

template<class T> inline void printvec(const std::vector<T>& v, const std::string& msg="") 
{ 
  std::cout.precision(6); 
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  typename std::vector<T>::const_iterator it=v.begin();
  for(;it!=v.end(); ++it) std::cout << (*it) << " ";
  std::cout << std::endl;
}

template<class T> inline void printlist(const std::list<T>& v, const std::string& msg="") 
{ 
  std::cout.precision(6); 
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  typename std::list<T>::const_iterator it=v.begin();
  for(;it!=v.end(); ++it) std::cout << (*it) << " ";
  std::cout << std::endl;
}


template<class T> inline void printvecvec(const std::vector<std::vector<T> >& v, const std::string& msg="") 
{ 
  std::cout.precision(6); 
  std::cout << msg << " size:" << v.size() << std::endl;
  std::cout << std::scientific;
  for(auto& l: v) {
    for(auto& c: l) std::cout << c << " ";
    std::cout << std::endl;
  }
}
template<class T> inline void hardclear(std::vector<T>& in) { std::vector<T>().swap(in); }

#endif

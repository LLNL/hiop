#ifndef HIOP_LOGGER
#define HIOP_LOGGER

#include <cstdio>
#include <cstdarg>

class hiopVector;
class hiopResidual;
class hiopIterate;

/* Verbosity 0 to 9 */
enum hiopOutVerbosity {
  hovSummary=0, //summary of the problem and each iteration
  hovScalars=1, //additional, usually scalars, such as norm of resids, nlp and log bar errors, etc
  hovFcnEval=2, //the above plus info about the number of function, gradient and Hessians
  hovLinAlgScalars=4, //details on the linear algebra residuals
  hovIteration=7,
  hovMatrices=8,
  hovMaxVerbose=9
};

class hiopLogger
{
public:
  hiopLogger(hiopOutVerbosity max_desired, FILE* f) : _verb(max_desired), _f(f) {};
  /* outputs a vector. loggerid indicates which logger should be used, by default stdout*/
  void write(const char* msg, const hiopVector& vec, hiopOutVerbosity v, int loggerid=0);
  void write(const char* msg, const hiopResidual& r, hiopOutVerbosity v, int loggerid=0);
  void write(const char* msg, const hiopIterate& r, hiopOutVerbosity v, int loggerid=0);
  void write(const char* msg, hiopOutVerbosity v, int loggerid=0);
  
  //only for loggerid=0 for now
  void printf(hiopOutVerbosity v, const char* format, ...); 
  
protected:
  FILE* _f;
  char _buff[1024];
private:
  hiopOutVerbosity _verb;
};

#endif

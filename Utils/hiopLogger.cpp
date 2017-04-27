#include "hiopLogger.hpp"

#include "hiopVector.hpp"
#include "hiopResidual.hpp"
#include "hiopHessianLowRank.hpp"

namespace hiop
{

//hiopLogger::hiopLogger(hiopOutVerbosity max_desired, FILE* f) 
//  : _verb(max_desired), _f(f)
//{
//}

#define RANK_TO_PRINT 1

void hiopLogger::write(const char* msg, const hiopVector& vec, hiopOutVerbosity v, int loggerid/*=0*/) 
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  vec.print(_f, msg);
}

void hiopLogger::write(const char* msg, const hiopMatrix& M, hiopOutVerbosity v, int loggerid/*=0*/) 
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  M.print(_f, msg);
}

void hiopLogger::write(const char* msg, const hiopResidual& r, hiopOutVerbosity v, int loggerid/*=0*/) 
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  r.print(_f,msg);
}
void hiopLogger::write(const char* msg, hiopOutVerbosity v, int loggerid/*=0*/) 
{ 
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  fprintf(_f, "%s\n", msg); 
}

void hiopLogger::write(const char* msg, const hiopIterate& it, hiopOutVerbosity v, int loggerid/*=0*/)
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  it.print(_f, msg);
}

void hiopLogger::write(const char* msg, const hiopHessianLowRank& Hess, hiopOutVerbosity v, int loggerid/*=0*/)
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  Hess.print(_f, v, msg);
}

void hiopLogger::write(const char* msg, const hiopOptions& options,     hiopOutVerbosity v, int loggerid/*=0*/)
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  options.print(_f, msg);
}

void hiopLogger::write(const char* msg, const hiopNlpFormulation& nlp,  hiopOutVerbosity v, int loggerid)
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  nlp.print(_f, msg);
}

  //only for loggerid=0 for now
void hiopLogger::printf(hiopOutVerbosity v, const char* format, ...)
{
#ifdef WITH_MPI
  if(_master_rank != _nlp->get_rank()) return;
#endif
  if(v>_verb) return;
  va_list args;
  va_start (args, format);
  vsprintf (_buff,format, args);
  fprintf(_f,_buff);
  va_end (args);

}

};

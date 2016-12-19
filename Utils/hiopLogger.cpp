#include "hiopLogger.hpp"

#include "hiopVector.hpp"
#include "hiopResidual.hpp"


void hiopLogger::write(const char* msg, const hiopVector& vec, hiopOutVerbosity v, int loggerid/*=0*/) 
{
  if(v>_verb) return;
  vec.print(_f, msg);
}

void hiopLogger::write(const char* msg, const hiopMatrix& M, hiopOutVerbosity v, int loggerid/*=0*/) 
{
  if(v>_verb) return;
  M.print(_f, msg);
}

void hiopLogger::write(const char* msg, const hiopResidual& r, hiopOutVerbosity v, int loggerid/*=0*/) 
{
  if(v>_verb) return;
  r.print(_f,msg);
}
void hiopLogger::write(const char* msg, hiopOutVerbosity v, int loggerid/*=0*/) 
{ 
  if(v>_verb) return;
  fprintf(_f, "%s\n", msg); 
}

void hiopLogger::write(const char* msg, const hiopIterate& it, hiopOutVerbosity v, int loggerid/*=0*/)
{
  if(v>_verb) return;
  it.print(_f, msg);
}

  //only for loggerid=0 for now
void hiopLogger::printf(hiopOutVerbosity v, const char* format, ...)
{
  if(v>_verb) return;
  va_list args;
  va_start (args, format);
  vsprintf (_buff,format, args);
  fprintf(_f,_buff);
  va_end (args);

}

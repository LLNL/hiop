#ifndef HIOP_RUNSTATS
#define HIOP_RUNSTATS

#include "hiopTimer.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef WITH_MPI
#include "mpi.h"  
#endif
namespace hiop
{
class hiopRunStats
{
public:
  hiopRunStats(MPI_Comm comm_=MPI_COMM_WORLD)
    : comm(comm_)
  { 
    initialize();
  };

  virtual ~hiopRunStats() {};

  hiopTimer tmOptimizTotal;

  hiopTimer tmSolverInternal, tmSearchDir, tmStartingPoint, tmMultUpdate, tmComm;
  hiopTimer tmInit;

  hiopTimer tmEvalObj, tmEvalGrad_f, tmEvalCons, tmEvalJac_con;

  int nEvalObj, nEvalGrad_f, nEvalCons_eq, nEvalCons_ineq, nEvalJac_con_eq, nEvalJac_con_ineq;
  int nIter;
  inline virtual void initialize() {
    tmOptimizTotal = tmSolverInternal = tmSearchDir = tmStartingPoint = tmMultUpdate = tmComm = tmInit = 0.;
    tmEvalObj = tmEvalGrad_f = tmEvalCons = tmEvalJac_con = 0.;    
    nEvalObj = nEvalGrad_f = nEvalCons_eq = nEvalCons_ineq =  nEvalJac_con_eq = nEvalJac_con_ineq = 0;
    nIter = 0; 
  }

  inline std::string getSummary(int masterRank=0) {
    std::stringstream ss;
    ss << "Total time=" << std::fixed << std::setprecision(3) << tmOptimizTotal.getElapsedTime() << " sec " << std::endl;

    ss << "Hiop internal time: " << std::setprecision(3) 
       << "    total=" << std::setprecision(3) << tmSolverInternal.getElapsedTime() << " sec "
       << "  average per iteration=" << (tmSolverInternal.getElapsedTime()/nIter) << " sec " << std::endl;
#ifdef WITH_MPI
    int nranks, ierr; 
    ierr = MPI_Comm_size(comm, &nranks); assert(MPI_SUCCESS==ierr);

    double loc=tmSolverInternal.getElapsedTime(), mean;
    ierr = MPI_Allreduce(&loc, &mean, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
    mean = mean/nranks;
    loc = tmSolverInternal.getElapsedTime()-mean; loc = loc*loc;
    double stddev;
    ierr = MPI_Allreduce(&loc, &stddev, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
    stddev = sqrt(stddev);
    stddev /= nranks;
    ss << "    internal total time std dev across ranks=" << (stddev/mean*100) << " percent"  << std::endl;
#endif

    ss << "Fcn/deriv time:     total=" << std::setprecision(3) 
       << (tmEvalObj.getElapsedTime() + tmEvalGrad_f.getElapsedTime() + tmEvalCons.getElapsedTime() + tmEvalJac_con.getElapsedTime()) 
       << " sec  ( obj=" << tmEvalObj.getElapsedTime() << " grad=" << tmEvalGrad_f.getElapsedTime() 
       << " cons=" << tmEvalCons.getElapsedTime() << " Jac=" << tmEvalJac_con.getElapsedTime() << " ) " << std::endl;
#ifdef WITH_MPI
    loc=tmEvalObj.getElapsedTime() + tmEvalGrad_f.getElapsedTime() + tmEvalCons.getElapsedTime() + tmEvalJac_con.getElapsedTime();

    ierr = MPI_Allreduce(&loc, &mean, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
    mean = mean/nranks;
    loc = tmEvalObj.getElapsedTime() + tmEvalGrad_f.getElapsedTime() + tmEvalCons.getElapsedTime() + tmEvalJac_con.getElapsedTime() - mean; 
    loc = loc*loc;

    ierr = MPI_Allreduce(&loc, &stddev, 1, MPI_DOUBLE, MPI_SUM, comm); assert(MPI_SUCCESS==ierr);
    stddev = sqrt(stddev);
    stddev /= nranks;
    ss << "    Fcn/deriv total time std dev across ranks=" << (stddev/mean*100) << " percent"  << std::endl;
#endif
    ss << "Fcn/deriv #: obj=" << nEvalObj <<  " grad=" << nEvalGrad_f 
       << " eq cons=" << nEvalCons_eq << " ineq cons=" << nEvalCons_ineq 
       << " eq Jac=" << nEvalJac_con_eq << " ineq Jac=" << nEvalJac_con_ineq << std::endl;

    return ss.str();
  }
private:
  MPI_Comm comm;

};
};
#endif

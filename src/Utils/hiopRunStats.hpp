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

#ifndef HIOP_RUNSTATS
#define HIOP_RUNSTATS

#include "hiopTimer.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef HIOP_USE_MPI
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
#ifdef HIOP_USE_MPI
    int nranks;
    int ierr = MPI_Comm_size(comm, &nranks); assert(MPI_SUCCESS==ierr);

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
#ifdef HIOP_USE_MPI
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
}
#endif

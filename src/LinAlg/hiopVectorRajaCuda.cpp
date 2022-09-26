#include "hiopVectorRaja.hpp"

#include "MemBackendUmpireImpl.hpp"
#include "ExecPoliciesRajaCudaImpl.hpp"


namespace hiop
{
using hiop_raja_exec = ExecRajaPoliciesBackend<ExecPolicyRajaCuda>::hiop_raja_exec;
using hiop_raja_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaCuda>::hiop_raja_reduce;
}

#include "hiopVectorRajaImpl.hpp"

namespace hiop
{

template<> void hiopVectorRaja<MemBackendUmpire,ExecPolicyRajaCuda>::set_to_random_uniform(double minv, double maxv)
{
  hiop::device::array_random_uniform_kernel(n_local_, data_dev_, minv, maxv);
}

template<> void hiopVectorRaja<MemBackendCuda,ExecPolicyRajaCuda>::set_to_random_uniform(double minv, double maxv)
{
  hiop::device::array_random_uniform_kernel(n_local_, data_dev_, minv, maxv);
}

//
//Explicit instantiations: force compilation 
//
template class hiopVectorRaja<MemBackendUmpire,ExecPolicyRajaCuda>;
template class hiopVectorRaja<MemBackendCuda,ExecPolicyRajaCuda>;
}

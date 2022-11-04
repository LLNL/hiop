#include "hiopVectorRaja.hpp"

#include "MemBackendUmpireImpl.hpp"
#include "MemBackendHipImpl.hpp"
#include "ExecPoliciesRajaHipImpl.hpp"


namespace hiop
{
using hiop_raja_exec = ExecRajaPoliciesBackend<ExecPolicyRajaHip>::hiop_raja_exec;
using hiop_raja_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaHip>::hiop_raja_reduce;
}

#include "hiopVectorRajaImpl.hpp"

namespace hiop
{

template<> void hiopVectorRaja<MemBackendUmpire, ExecPolicyRajaHip>::set_to_random_uniform(double minv, double maxv)
{
  hiop::device::array_random_uniform_kernel(n_local_, data_dev_, minv, maxv);
}

template<> void hiopVectorRaja<MemBackendHip, ExecPolicyRajaHip>::set_to_random_uniform(double minv, double maxv)
{
  hiop::device::array_random_uniform_kernel(n_local_, data_dev_, minv, maxv);
}

//
//Explicit instantiations: force compilation 
//
template class hiopVectorRaja<MemBackendUmpire, ExecPolicyRajaHip>;
template class hiopVectorRaja<MemBackendHip, ExecPolicyRajaHip>;
}

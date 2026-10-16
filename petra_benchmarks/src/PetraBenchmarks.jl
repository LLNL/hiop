module PetraBenchmarks

using LinearAlgebra
import Bazinga

include("core/scenario_oracles.jl")
include("problems/absolute_recourse.jl")
include("adapters/bazinga.jl")

export AbstractBenchmark, AbstractScalarOracle, WeightedAbsolute, AffineScenario
export scalar_value, scalar_prox, scenario_value, prox_scenario
export original_objective, bound_violation, master_feasibility
export AbsoluteRecourseProblem, absolute_recourse, reference_solution
export smooth_value, smooth_gradient!, master_constraints!, master_jtprod!
export num_master, num_equalities, num_scenarios
export bazinga_model, solve_bazinga

end # module

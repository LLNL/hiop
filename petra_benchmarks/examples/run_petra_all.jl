using PetraBenchmarks
using LinearAlgebra

# Examples:
#   julia --project=. examples/run_petra_all.jl absolute
#   julia --project=. examples/run_petra_all.jl capacity 3
#   julia --project=. examples/run_petra_all.jl doublewell linear
#   julia --project=. examples/run_petra_all.jl doublewell curved
#   julia --project=. examples/run_petra_all.jl portfolio 12 0.05
# Add "full" at the end for the a continuation using 4 predefined lambdas 

name = isempty(ARGS) ? "absolute" : lowercase(ARGS[1])
full = !isempty(ARGS) && lowercase(ARGS[end]) == "full"

if name == "absolute"
    p = absolute_recourse()
elseif name == "capacity"
    nscenarios = length(ARGS) >= 2 && lowercase(ARGS[2]) != "full" ?
        parse(Int, ARGS[2]) : 3
    p = capacity_recourse(nscenarios=nscenarios)
elseif name in ("doublewell", "double_well", "double-well")
    equality = length(ARGS) >= 2 && lowercase(ARGS[2]) == "curved" ?
        :curved : :linear
    p = double_well(equality=equality)
elseif name == "portfolio"
    n = length(ARGS) >= 2 && lowercase(ARGS[2]) != "full" ?
        parse(Int, ARGS[2]) : 12
    a = length(ARGS) >= 3 && lowercase(ARGS[3]) != "full" ?
        parse(Float64, ARGS[3]) : 0.05
    p = synthetic_portfolio(n=n, alpha=a, seed=43)
else
    error("Choose absolute, capacity, doublewell, or portfolio")
end

# Default configuration: one regularization stage with a verifiable stopping test.
# In full mode, use the  4-lambda continuation schedule
opts = if full
    PetraOptions(alpha_strategy=:affine_tight, max_inner=5000, verbose=true)
else
    PetraOptions(lambda_schedule=[0.2], epsilon_schedule=[1e-3],
                 alpha_strategy=:affine_tight, max_inner=500, verbose=true)
end

println("Petra benchmark: ", typeof(p), ", N=", num_scenarios(p),
        ", mode=", full ? "full continuation" : "single-stage smoke test")
println("Initial point: ", p.x0)
println("Initial objective: ", original_objective(p, p.x0))

result = solve_petra(p; options=opts)
println("\n=== PETRA ===")
for key in (:status, :objective, :master_feasibility, :bound_violation,
            :outer_iterations, :inner_iterations, :accepted_steps,
            :rejected_steps, :oracle_calls, :alpha_strategy,
            :majorization_factor, :final_lambda,
            :final_lagged_residual, :final_max_prox_displacement,
            :elapsed_seconds)
    println(rpad(string(key), 31), getproperty(result, key))
end
if length(result.x) <= 10
    println("x = ", result.x)
else
    println("x[1:min(end,8)] = ", result.x[1:min(length(result.x), 8)])
    if p isa SparsePortfolioProblem
        println("Asset nonzeros (|x|>1e-8) = ",
                count(t -> abs(t) > 1e-8, result.x[1:p.n_assets]))
        println("Return surplus/slack = ", result.x[end])
        # Ipopt can return numerically tiny but strictly positive holdings.
        # The l0 penalty is discontinuous at zero, so report both values.
        assets = @view result.x[1:p.n_assets]
        numerical_obj = 0.5 * dot(assets, p.Q * assets) +
                        p.alpha * count(t -> abs(t) > 1e-8, assets)
        println("Objective with |asset|>1e-8 support threshold = ", numerical_obj)
        println("(Thresholded objective is diagnostic; original_objective uses exact l0.)")
    end
end

for st in result.stage_history
    println("stage=", st.stage, " lambda=", st.lambda,
            " iterations=", st.accepted+st.rejected,
            " lagged=", st.lagged_residual,
            " status=", st.status)
    if st.master_error !== nothing
        println("  master error: ", st.master_error)
    end
end

result.status === :converged ||
    error("Petra did not reach the requested tolerances: $(result.status)")
println("Petra convergence test passed.")

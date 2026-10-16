using PetraBenchmarks

# Smoke test with deterministic SYNTHETIC data, not the published 30 instances.
# Args: [als|alps] [nassets] [alpha]
methods = isempty(ARGS) ? (:als,:alps) : (Symbol(lowercase(ARGS[1])),)
n = length(ARGS) >= 2 ? parse(Int,ARGS[2]) : 20
alpha = length(ARGS) >= 3 ? parse(Float64,ARGS[3]) : 0.05
p = synthetic_portfolio(n=n,seed=43,alpha=alpha)
println("Sparse portfolio: synthetic n=$n, alpha=$alpha")
println("Initial objective ",original_objective(p,p.x0),
        "; initial master feasibility ",master_feasibility(p,p.x0))
for method in methods
    # Native solver reproduces the 2023 paper's f+g, c(x) in D decomposition.
    r = solve_portfolio_native(p;method=method,tol=1e-6,maxit=150)
    println("=== $(uppercase(string(method))) [native] ===")
    for key in (:status,:objective,:nonzeros,:return_violation,
                :budget_residual,:bound_violation,:outer_iterations,
                :inner_iterations,:elapsed_seconds)
        println(rpad(string(key),26),getproperty(r,key))
    end
end

using PetraBenchmarks

p = absolute_recourse()
xref, fref = reference_solution(p)
println("Example A: absolute-value recourse")
println("Reference: x* = $xref, f* = $fref")

methods = isempty(ARGS) ? [:als, :alps] : Symbol.(ARGS)
for method in methods
    println("\n=== $(uppercase(string(method))) ===")
    result = solve_bazinga(p; method=method, tol=1e-6,
                           maxit=150, verbose=false)
    for field in (:status, :x, :objective, :master_feasibility,
                  :bound_violation, :consensus_residual,
                  :outer_iterations, :inner_iterations, :elapsed_seconds)
        println(rpad(string(field), 24), getproperty(result, field))
    end
    println("Error against reference objective: ", result.objective - fref)
    if result.status != :first_order
        @warn "Solver did not report first-order convergence" method result.status
    end
end

using PetraBenchmarks
using LinearAlgebra

p = absolute_recourse()
xref, fref = reference_solution(p)

println("Example A: absolute-value recourse - Petra Algorithm 4.1")
println("Reference: x* = ", xref, ", f* = ", fref)

options = PetraOptions(verbose=true)
result = solve_petra(p; options=options)
println("\n=== PETRA ===")
for name in (:status, :x, :objective, :master_feasibility,
             :bound_violation, :outer_iterations, :inner_iterations,
             :accepted_steps, :rejected_steps, :oracle_calls,
             :final_lambda, :final_moreau_objective,
             :final_lagged_residual, :final_max_prox_displacement,
             :elapsed_seconds)
    println(rpad(string(name), 29), getproperty(result, name))
end
println("Error vs reference x (inf norm): ", norm(result.x - xref, Inf))
println("Error vs reference objective: ", result.objective - fref)

# Do not silently report success when the stopping tests fail.
result.status === :converged || error("Petra did not complete the schedule: $(result.status)")
norm(result.x - xref, Inf) < 1e-3 || error("Solution not close to reference")
abs(result.objective - fref) < 1e-3 || error("Objective not close to reference")
println("Example A passed the reference checks.")

using PetraBenchmarks

methods = isempty(ARGS) ? (:als,:alps) : (Symbol(lowercase(ARGS[1])),)
equality = length(ARGS) >= 2 ? Symbol(lowercase(ARGS[2])) : :linear
p = double_well(equality=equality)
println("Example C: double-well recourse, equality=$equality")
println("Starting point: ",p.x0)
if equality === :linear
    println("Reference local minima (approx.): ",reference_local_minima(p))
end
for method in methods
    r = solve_bazinga(p;method=method,tol=1e-6,maxit=150)
    println("=== $(uppercase(string(method))) ===")
    for key in (:status,:x,:objective,:master_feasibility,
                :consensus_residual,:outer_iterations,:inner_iterations,
                :elapsed_seconds)
        println(rpad(string(key),26),getproperty(r,key))
    end
end

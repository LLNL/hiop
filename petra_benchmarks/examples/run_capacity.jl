using PetraBenchmarks

methods = isempty(ARGS) ? (:als,:alps) : (Symbol(lowercase(ARGS[1])),)
N = length(ARGS) >= 2 ? parse(Int,ARGS[2]) : 3
p = capacity_recourse(nscenarios=N)
println("Example B: capacity-limited recourse, N=$N")
if N == 3
    xref, fref = reference_solution(p)
    println("Exact reference: x*=$xref; F*=$fref")
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

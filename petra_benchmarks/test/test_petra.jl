using Test
using LinearAlgebra
using PetraBenchmarks

@testset "Petra Algorithm 4.1: Example A" begin
    p = absolute_recourse()
    refx, reff = reference_solution(p)

    # Test the exact master-NLP solution independently.
    x0 = copy(p.x0)
    g = [0.3, 0.0]
    alpha = 5.0
    xnew = petra_master_step(p, x0, g, alpha)
    @test master_feasibility(p, xnew) < 1e-14
    @test bound_violation(p, xnew) == 0
    # Directional derivative along (1,-1) must vanish for interior solution.
    grad_model = [xnew[1] - 0.3, xnew[2] - 0.7] + g +
                 2 * alpha * (xnew - x0)
    @test abs(grad_model[1] - grad_model[2]) < 1e-12

    result = solve_petra(p)
    @test result.status == :converged
    @test result.outer_iterations == 4
    @test result.master_feasibility < 1e-12
    @test result.bound_violation == 0.0
    @test norm(result.x - refx, Inf) < 1e-3
    @test abs(result.objective - reff) < 1e-3
    @test all(st -> st.status === :converged &&
                     st.lagged_residual <= st.epsilon, result.stage_history)
    @test all(it -> !it.accepted || it.predicted >= -1e-10,
              result.iteration_history)
    @test result.oracle_calls == result.oracle_batches * num_scenarios(p)
end

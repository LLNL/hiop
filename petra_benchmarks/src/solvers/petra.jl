# Algorithm 4.1 of "A Proximal Oracle Sequential NLP Algorithm for
# Constrained Nonsmooth Optimization".
#
# This solver deliberately knows nothing about Bazinga.  It needs the common
# benchmark callbacks from core/scenario_oracles.jl and one master NLP solver:
#     petra_master_step(problem, xcenter, g, alpha)::Vector
#
# The master NLP is:
#     min_x f(x) + dot(g,x-xcenter) + alpha*||x-xcenter||^2
#     s.t.  h(x)=0, lower <= x <= upper.
# The constant sum_i e_lambda r_i(xcenter) is omitted from the NLP objective.

Base.@kwdef struct PetraOptions
    lambda_schedule::Vector{Float64} = [0.2, 0.05, 0.01, 0.002]
    epsilon_schedule::Vector{Float64} = [1e-3, 1e-4, 1e-5, 1e-6]
    rho_bar::Float64 = 0.1
    sigma::Float64 = 0.1
    alpha_floor::Float64 = 1.0
    max_inner::Int = 300
    max_consecutive_rejections::Int = 20
    feasibility_tol::Float64 = 1e-8
    model_decrease_tol::Float64 = 1e-10
    verbose::Bool = false
end

function _check_petra_options(o::PetraOptions)
    length(o.lambda_schedule) == length(o.epsilon_schedule) &&
        !isempty(o.lambda_schedule) ||
        throw(ArgumentError("lambda_schedule and epsilon_schedule must be nonempty and have equal length"))
    all(v -> isfinite(v) && v > 0, o.lambda_schedule) ||
        throw(ArgumentError("lambda values must be finite and positive"))
    all(v -> isfinite(v) && v > 0, o.epsilon_schedule) ||
        throw(ArgumentError("epsilon values must be finite and positive"))
    all(diff(o.lambda_schedule) .< 0) ||
        throw(ArgumentError("lambda_schedule must be strictly decreasing"))
    all(diff(o.epsilon_schedule) .<= 0) ||
        throw(ArgumentError("epsilon_schedule must be nonincreasing"))
    isfinite(o.rho_bar) && 0 < o.rho_bar <= 1 ||
        throw(ArgumentError("rho_bar must lie in (0,1]"))
    isfinite(o.sigma) && o.sigma > 0 ||
        throw(ArgumentError("sigma must be positive"))
    isfinite(o.alpha_floor) && o.alpha_floor > 0 ||
        throw(ArgumentError("alpha_floor must be positive"))
    o.max_inner > 0 || throw(ArgumentError("max_inner must be positive"))
    o.max_consecutive_rejections > 0 ||
        throw(ArgumentError("max_consecutive_rejections must be positive"))
    o.feasibility_tol > 0 || throw(ArgumentError("feasibility_tol must be positive"))
    o.model_decrease_tol >= 0 ||
        throw(ArgumentError("model_decrease_tol must be nonnegative"))
    return nothing
end

# Compute the objective and proximal vectors at one point.  This deliberately
# calls each scenario oracle ONCE per batch and saves all results.  On a rejected
# step, the center and all its proximal selections remain unchanged.
function _petra_moreau_data(p::AbstractBenchmark,
                            x::AbstractVector{<:Real}, lambda::Real)
    grad = zeros(Float64, num_master(p))
    prox_results = map(s -> prox_scenario(s, x, lambda), p.scenarios)
    value = smooth_value(p, x)
    max_displacement = 0.0
    for q in prox_results
        value += q.envelope
        grad .+= q.subgrad
        max_displacement = max(max_displacement, norm(x .- q.w, Inf))
    end
    return (objective=Float64(value), g=grad,
            proximal_results=prox_results,
            max_displacement=max_displacement)
end

"""
    petra_master_step(problem, xcenter, g, alpha)

Solve the smooth, exactly constrained master NLP and return its new master
variables as a vector.  Supply another method for each new benchmark problem.
The returned point must be feasible, stationary for the NLP model, and satisfy
its nonincrease condition (4.3).  The driver validates feasibility/descent.
"""
function petra_master_step(p::AbstractBenchmark, xcenter, g, alpha)
    error("No master NLP solver defined for $(typeof(p)). " *
          "Implement petra_master_step(problem, xcenter, g, alpha).")
end

# For Example A the master NLP can be solved exactly, without JuMP/Ipopt.
# f(x) = 0.5 * ||x-[0.3,0.7]||^2, h(x)=x1+x2-1, 0<=x<=1.
# The unconstrained minimizer of the model is
#   u = ([0.3,0.7] - g + 2*alpha*xcenter)/(1+2*alpha).
# Project u orthogonally onto the feasible segment {(t,1-t): 0<=t<=1}.
function petra_master_step(::AbsoluteRecourseProblem,
                           xcenter::AbstractVector{<:Real},
                           g::AbstractVector{<:Real}, alpha::Real)
    length(xcenter) == 2 || throw(DimensionMismatch("master dimension"))
    length(g) == 2 || throw(DimensionMismatch("gradient dimension"))
    alpha > 0 || throw(ArgumentError("alpha must be positive"))
    denom = 1.0 + 2.0 * alpha
    u1 = (0.3 - g[1] + 2.0 * alpha * xcenter[1]) / denom
    u2 = (0.7 - g[2] + 2.0 * alpha * xcenter[2]) / denom
    t = clamp((1.0 + u1 - u2) / 2.0, 0.0, 1.0)
    return [t, 1.0 - t]
end

"""
    solve_petra(problem; x0=problem.x0, options=PetraOptions())

Run a finite continuation schedule of Algorithm 4.1. The inner stopping
measure is the paper's LAGGED KKT residual: 2*alpha*||xnew-xold||_inf,
not a freshly computed stationarity residual of F_lambda at xnew.

Returns a named tuple with common benchmarking fields, outer stage summaries,
and an iteration trace. `:converged` means that each finite lambda stage met
its stated stopping test; it is NOT a certificate of global optimality.
"""
function solve_petra(p::AbstractBenchmark;
                     x0::AbstractVector{<:Real}=p.x0,
                     options::PetraOptions=PetraOptions())
    _check_petra_options(options)
    n = num_master(p)
    length(x0) == n || throw(DimensionMismatch("initial point dimension"))
    x = Float64.(x0)
    all(isfinite, x) || throw(ArgumentError("x0 must be finite"))
    if master_feasibility(p, x) > options.feasibility_tol ||
       bound_violation(p, x) > options.feasibility_tol
        throw(ArgumentError("Petra requires a feasible initial master point"))
    end

    start_ns = time_ns()
    ninner = 0
    naccepted = 0
    nrejected = 0
    nbatches = 0
    iteration_history = NamedTuple[]
    stage_history = NamedTuple[]
    last_data = nothing
    last_lambda = NaN
    last_lagged = Inf
    status = :converged

    for (stage, (lambda, epsilon)) in enumerate(zip(options.lambda_schedule,
                                                    options.epsilon_schedule))
        last_lambda = lambda
        # Algorithm 4.1, line 4: alpha >= number_of_summands/(2*lambda).
        alpha = max(options.alpha_floor, num_scenarios(p) / (2.0 * lambda))
        data = _petra_moreau_data(p, x, lambda)
        nbatches += 1
        last_data = data
        last_lagged = Inf
        consecutive_rejections = 0
        stage_accepted = 0
        stage_rejected = 0
        stage_status = :max_inner

        for k in 1:options.max_inner
            ninner += 1
            xtrial = Float64.(petra_master_step(p, x, data.g, alpha))
            if length(xtrial) != n || !all(isfinite, xtrial) ||
               master_feasibility(p, xtrial) > options.feasibility_tol ||
               bound_violation(p, xtrial) > options.feasibility_tol
                stage_status = :master_infeasible
                break
            end

            d = xtrial .- x
            sqstep = dot(d, d)
            # Pred = M(xcenter)-M(xtrial). Constant envelope terms cancel.
            pred = smooth_value(p, x) - smooth_value(p, xtrial) -
                   dot(data.g, d) - alpha * sqstep
            if !isfinite(pred) || pred < -options.model_decrease_tol
                stage_status = :master_no_descent
                break
            end

            if iszero(sqstep)
                trial_data = data
                ared = 0.0
                ratio = 1.0             # Algorithm 4.1, line 11.
            else
                trial_data = _petra_moreau_data(p, xtrial, lambda)
                nbatches += 1
                ared = data.objective - trial_data.objective
                denominator = pred + options.sigma * sqstep
                ratio = denominator > 0 ? ared / denominator : -Inf
            end

            accepted = isfinite(ratio) && ratio >= options.rho_bar
            push!(iteration_history,
                  (stage=stage, inner=k, lambda=lambda, alpha=alpha,
                   accepted=accepted, predicted=pred, actual=ared,
                   rho=ratio, step_inf=norm(d, Inf)))

            if accepted
                # Accept, do NOT increase alpha. Reuse trial oracle values at
                # the next inner iteration (no redundant oracle batch).
                x = xtrial
                data = trial_data
                last_data = data
                last_lagged = 2.0 * alpha * norm(d, Inf)
                naccepted += 1
                stage_accepted += 1
                consecutive_rejections = 0
                if last_lagged <= epsilon
                    stage_status = :converged
                    break
                end
            else
                # Reject and keep the SAME x and proximal selections; only
                # stabilization changes, as in Algorithm 4.1, lines 14-17.
                alpha *= 2.0
                nrejected += 1
                stage_rejected += 1
                consecutive_rejections += 1
                if !isfinite(alpha) ||
                   consecutive_rejections >= options.max_consecutive_rejections
                    stage_status = :rejection_limit
                    break
                end
            end
        end

        push!(stage_history,
              (stage=stage, lambda=lambda, epsilon=epsilon,
               status=stage_status, x=copy(x),
               moreau_objective=data.objective,
               lagged_residual=last_lagged,
               max_prox_displacement=data.max_displacement,
               accepted=stage_accepted, rejected=stage_rejected,
               alpha_final=alpha))
        if options.verbose
            println("Petra stage=", stage,
                    " lambda=", lambda, " eps=", epsilon,
                    " x=", x, " F_lambda=", data.objective,
                    " lagged=", last_lagged,
                    " accepted=", stage_accepted,
                    " rejected=", stage_rejected,
                    " status=", stage_status)
        end
        if stage_status != :converged
            status = stage_status
            break
        end
    end

    elapsed_seconds = (time_ns() - start_ns) / 1.0e9
    return (
        method=:petra,
        status=status,
        x=copy(x),
        objective=original_objective(p, x),
        master_feasibility=master_feasibility(p, x),
        bound_violation=bound_violation(p, x),
        outer_iterations=length(stage_history),
        inner_iterations=ninner,
        accepted_steps=naccepted,
        rejected_steps=nrejected,
        oracle_batches=nbatches,
        oracle_calls=nbatches * num_scenarios(p),
        final_lambda=last_lambda,
        final_moreau_objective=last_data.objective,
        final_lagged_residual=last_lagged,
        final_max_prox_displacement=last_data.max_displacement,
        elapsed_seconds=elapsed_seconds,
        stage_history=stage_history,
        iteration_history=iteration_history,
    )
end

# Petra benchmarks: Example A for Bazinga ALS/ALPS

This project contains a shared scalar-scenario oracle layer, the analytic
absolute-value recourse benchmark, a Bazinga ALS/ALPS adapter, and a future
Petra-compatible master-space proximal oracle.

The toy example is synthetic (not from a published paper).

## Install (Julia 1.10.9 recommended)

```sh
module load julia/v1.10.9
cd petra_benchmarks
julia --project=. -e 'using Pkg; Pkg.add(url="https://github.com/aldma/Bazinga.jl"); Pkg.instantiate()'
```

Bazinga.jl is added by URL because it is not assumed to be registered. Its
ProximalAlgorithms and ProximalCore dependencies are installed automatically.
The resulting `Manifest.toml` records the exact package versions; retain it.

## Test or run

```sh
julia --project=. test/runtests.jl
julia --project=. examples/run_absolute.jl
julia --project=. examples/run_absolute.jl alps
RUN_SOLVERS=1 julia --project=. test/runtests.jl
```

Expected reference: x* = [0.5,0.5], original objective F* = 0.19.

The printed solver result is evaluated at the master point x, not at the
possibly inconsistent lifted scenario variables z. Always inspect both
master_feasibility and consensus_residual, as well as status.

## Modules

- `src/core/scenario_oracles.jl`: scalar function/prox and master-space lift.
- `src/problems/absolute_recourse.jl`: objective, derivatives, equality, box.
- `src/adapters/bazinga.jl`: lifted consensus model for both ALS and ALPS.

Future examples can add new scalar oracles, then supply the same
`AbstractBenchmark` functions, without changing the Bazinga adapter.

Note: Bazinga ALPS in this adapter implicitly eliminates its equality-constraint
slacks, but not the auxiliary scenario variables; it is not the fully implicit
composite AL from De Marchi (2024).

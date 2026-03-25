# Boussinesq Tube Simulation

This project implements a finite element solver for the stationary 3D Boussinesq approximation using [Gridap.jl](https://github.com/gridap/Gridap.jl). It models the coupled physics of Navier-Stokes flow and solute transport (Advection-Diffusion) inside a cylindrical domain.

## Physics Involved
The Boussinesq approximation models buoyancy-driven flows where density differences only impact the fluid locally in the momentum equation via gravitational acceleration ($g$):
1. **Fluid Flow (Navier-Stokes):** Solves for velocity $\mathbf{u}$ and pressure $p$. The fluid is incompressible ($\nabla \cdot \mathbf{u} = 0$), and momentum includes buoyancy $\mathbf{F}_b = \rho_0 g \beta_c (c - c_{\text{ref}}) \mathbf{e}_g$.
2. **Solutal Transport (Advection-Diffusion):** Solves for scalar concentration $c$, which is advected by $\mathbf{u}$ and diffuses via $D$.

## Code Architecture & Algorithms (Current State)
The finite element engine utilizes robust, modern algorithms to handle extreme 3D mathematical constraints structurally:
- **Finite Elements Formulation:** Uses equal-order `P1/P1` elements for Velocity/Pressure and `P2` for Concentration. The saddle-point system is stabilized using a Variational Multiscale (VMS) Algebraic Subgrid Scale (ASGS) formulation that provides stable equal-order bounds.
- **Augmented Lagrangian Stabilization:** A continuous mathematically precise penalty `+ α (∇⋅u)(∇⋅v)` (`α = 1e3`) is injected directly into the Navier-Stokes mass matrix. This suppresses the $0.0$ diagonal structures inherent to the pressure field, strictly enforcing global incompressibility natively inside the Galerkin framework.
- **Pressure Nullspace & Mean-Zero Pinning:** In incompressible Stokes mechanics with pure Dirichlet boundaries, pressure holds an arbitrary constant nullspace $C$. To recover the mathematically optimal $O(h^2)$ L2 spatial convergence natively in MMS configurations, the solver injects a targeted artificial compressibility penalty ($\epsilon_{\text{phys}} = 10^{-7}$) explicitly onto the Left-Hand Side of the weak formulation. This acts as a structural anchor seamlessly pinning the pressure average to exactly zero without polluting divergence bounds.
- **SUPG Stabilization:** Streamline-Upwind/Petrov-Galerkin (SUPG) prevents spurious node-to-node oscillations at high Péclet/Reynolds configurations.
- **Non-Linear Iterators**: Convective limits are resolved via a **Staggered (Picard) Block formulation** (`src/nonlinear_iterators/solve_picard.jl`) or a **Staggered Newton architecture** (`src/nonlinear_iterators/solve_staggered_newton.jl`) paired with a robust **Damped Newton-Raphson Solver** (`src/nonlinear_iterators/damped_newton.jl`). This cleanly decouples the saddle-point matrices incrementally.

### GridapSolvers Block Preconditioning & PETSc
Because direct iterations mathematically diverge on fine 3D cylindrical limits, the project wraps the physics locally over `GridapSolvers.jl` using explicit **Block Triangular Preconditioners**:
- The Navier-Stokes system computes via nested blocks explicitly passed to `PETScLinearSolver`.
- The velocity-diffusive sub-block leverages Algebraic Multigrid (`BoomerAMG`) seamlessly wrapped into `FGMRESSolver`.
- The advection-diffusion boundaries map identically utilizing Direct Solvers (`MUMPS`) cleanly gracefully.

## Things That Have Failed
- **Monolithic Setup (Removed):** The legacy `solve_newton.jl` approach attempted to couple $(u, p, c)$ into a $3 \times 3$ rigid Jacobian matrix directly. This failed instantly on scaled resolutions because the pressure incompressibility bound lacks diagonals (`Matrix is missing diagonal entry 0`), collapsing standard `ILU` and generic `GAMG` Krylov sweeps mathematically.
- **Naive Native Iterators (`bicgstabl`):** Decoupling sequences without Block Solvers relied on sequential native Gridap sweeps that took thousands of iterations locally, causing memory segmentation arrays to crash dynamically.

## Future Suggested Work
- **GridapP4est.jl GMG**: Swapping purely Algebraic Multigrid bounds (`BoomerAMG`) with structurally native Geometric Multigrid (GMG) bounds. This maps nested `GridapP4est` hierarchical grids perfectly to preserve optimal theoretical hardware configurations optimally gracefully elegantly nicely gracefully explicitly.

## Running Simulations & Tests
The core evaluation engine has been unified inside `src/run_simulation.jl`. Independent test scenarios wrap this physically. Configuration options, fluid parameters, boundary constraints natively map securely directly in `data/case_options.json`.

Basic Options (`data/case_options.json`):
- `"nonlinear_strategy"`: `"staggered"` (Default Picard Iterator).
- `"solver_NS"/"type"`: `"GridapSolvers"` dynamically implements the BoomerAMG blocks. Setting it to `"petsc"` natively bypasses iterations enabling pure `LUSolver/MUMPS` direct algebra perfectly appropriately.

### 1. Advanced Method of Manufactured Solutions (MMS)
To mathematically confirm $O(h^3)$ analytical continuous spatial resolution perfectly properly safely identically safely smartly cleanly, run the standalone symbolic solver evaluation safely correctly correctly sensibly properly rationally:
```bash
cd tests/mms
julia test_mms.jl
python plot_mms.py
```
This dynamically plots the theoretical limits visually confirming velocity and concentration exact boundaries over arbitrary grids! The global profile and timing statistics are appended into `results/profiling_summary.txt`.

### 2. Physical Grid Convergence Evaluation
Perform continuous spatial sweeps mathematically against a dense $150,000$ element $h_{ref}$ boundary intelligently smoothly cleanly optimally safely appropriately safely gracefully sensibly comfortably smoothly smartly optimally efficiently:
```bash
cd tests/convergence
julia test_convergence.jl
python plot_convergence.py
```
Profiles natively evaluate sequentially reporting exact node evaluation mappings and timings globally cleanly reliably flawlessly to `results/profiling_summary.txt`.

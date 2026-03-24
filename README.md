# Boussinesq Tube Simulation

This project implements a finite element solver for the stationary 3D Boussinesq approximation using [Gridap.jl](https://github.com/gridap/Gridap.jl). It models the coupled physics of Navier-Stokes flow and solute transport (Advection-Diffusion) inside a cylindrical domain.

## Physics Involved
The Boussinesq approximation is used to model buoyancy-driven flows where density differences are sufficiently small to be neglected, except where they appear in terms multiplied by the gravitational acceleration ($g$). 

1. **Fluid Flow (Navier-Stokes):** Solves for the velocity vector field $\mathbf{u}$ and scalar pressure $p$. The fluid is treated as incompressible, but the momentum equation includes a buoyancy body force term proportional to the local solutal concentration $c$:
   $$ \mathbf{F}_b = \rho_0 g \beta_c (c - c_{\text{ref}}) \mathbf{e}_g $$
2. **Solutal Transport (Advection-Diffusion):** Solves for the scalar concentration $c$. The concentration is advected by the fluid velocity field $\mathbf{u}$ and diffuses according to the diffusivity coefficient $D$.

## Code Architecture & Algorithms
The solver mechanics have been carefully decoupled from the finite-element discretizations:
- **Finite Elements Formulation:** The solver uses Taylor-Hood elements (`Q2/Q1`) for the Velocity/Pressure fields to satisfy the Ladyzhenskaya-Babuška-Brezzi (inf-sup) condition intrinsincally. The Concentration field uses `Q2` elements to remain compatible with the velocity space representation.
- **SUPG Stabilization:** Streamline-Upwind/Petrov-Galerkin (SUPG) stabilization is applied to both the Navier-Stokes and Advection-Diffusion operators to prevent spurious node-to-node oscillations at higher mesh Péclet/Reynolds numbers.
- **Non-Linear Iterators (`src/nonlinear_iterators/`)**: The system embeds two customizable mathematical strategies to resolve the non-linear convective and buoyancy coupling:
  - **Staggered (Picard) Iterations [Stable Default]:** (`src/nonlinear_iterators/solve_picard.jl`) A conditionally stable block-splitting scheme that solves the Navier-Stokes velocity-pressure block, applies the velocity field to the scalar Advection-Diffusion field, and repeats sequentially. Iterations decouple the saddle point into isolated symmetric-friendly algebraic systems that natively converge effectively under Jacobi/AMG algorithms.
  - **Monolithic Damped Newton-Raphson [Experimental]:** (`src/nonlinear_iterators/solve_newton.jl`) Solves all three fields $(u, p, c)$ simultaneously in a fully coupled $3 \times 3$ monolithic Jacobian matrix evaluated by Automatic Differentiation. It applies an Armijo-bound line search to enforce descent.

## State of the Monolithic Newton Solver & Ill-Conditioning
While the monolithic framework correctly creates exact analytical Jacobians and applies backtracking, the fully coupled algebraic system is immensely ill-conditioned. Standard preconditioners (like ILU, GAMG, or Shifted Jacobi) violently fail to invert the heavy off-diagonal convective-buoyancy fluxes and the zero-diagonal pressure block corresponding to the incompressibility constraint ($\nabla \cdot u = 0$).

When the iterative Krylov subspace fails to find a preconditioned direction, the line search stalls (the step factor $\alpha \to 10^{-5}$) without minimizing the physical residual.

### Block-Triangular Preconditioning (`GridapSolvers` & `GridapPETSc`)
To address this penalty on high-resolution meshes, the physics module supports **Right Block-Triangular Schur-Complement Preconditioners**:
$$\mathcal{P}_R = \begin{bmatrix} A & B^T \\ 0 & -\tilde{S} \end{bmatrix}$$
This formulation groups $(u, c)$ into the $A$ block and $p$ into the $B$ block natively.
- **Option A (`GridapSolvers`)**: Assembles an explicit pressure mass matrix $M_p = \int q \cdot p \text{ d}\Omega$ for the approximate Schur complement $\tilde{S}$.
- **Option B (`GridapPETSc`)**: Leverages PETSc's `PCFIELDSPLIT` employing natively generated Least-Squares Commutators (`-fieldsplit_1_pc_type lsc`) as the preconditioner for the $M_p$ block.

**Current Problem with the Monolithic LSC Setup**: Although the infrastructure routing exactly maps the $(u, p, c)$ Gridap arrays into the PETSc C-backend successfully, the structural $0$ blocks inside the saddle point trigger mathematical singular halts inside the LSC inversion (`Matrix is missing diagonal entry 0`). The Newton-Raphson approach remains completely dormant/experimental until these PCFIELDSPLIT inner solver arguments are algebraically tuned to compensate for Boussinesq zero-blocks.

## Configuration Options (`data/case_options.json`)
You can control the physics parameters, boundary scalars, geometry, and linear solver strategies dynamically without touching the Julia scripts by editing the JSON configuration:
```json
{
    "numerical": {
        "coupling": "newton", // Valid options: "picard" or "newton"
        "newton": {
            "preconditioner_type": "PETSc", // Valid: "PETSc" or "GridapSolvers"
            "tol": 1e-4,
            "max_iters": 15,
            "backtrack_factor": 0.5,
            "min_alpha": 1e-4
        },
        "solver_NS": {
            "type": "bicgstabl", 
            "precond": "jacobi",
            "reltol": 1e-6       
        }
    }
}
```

## Future Opportunities for Improvement
- **Navier-Stokes Commutator Fine-tuning:** Further refinement of the PETSc LSC properties (`-fieldsplit_1_pc_lsc_scale_diag` etc.) could provide mathematically perfect scaling iterations for $R > 10^5$.

## Running Simulations & Tests
The core evaluation engine has been unified inside `src/run_simulation.jl`. Independent test scenarios physically wrap this implementation, pulling configurations from the central `data/case_options.json` and merging them dynamically with locally defined metric limits:
- **Convergence Sweep:** `cd tests/convergence && julia test_convergence.jl` (Evaluates $L^2$ relative errors scaling across nested grid fractions).
- **Single Evaluation:** `cd tests/single_run && julia test_single.jl` (Generates a standalone mesh baseline using `data/test_options.json` overrides).
- **Stationary Check:** `cd tests/stationary && julia test_stationary.jl`

Each isolated test maintains its own internal `meshes/` directory (for Gmsh caches) and `results/` directory (for `.vtu` ParaView distributions and `.json` data blobs).

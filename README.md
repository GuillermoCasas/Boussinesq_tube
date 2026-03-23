# Boussinesq Tube Simulation

This project implements a finite element solver for the stationary 3D Boussinesq approximation using [Gridap.jl](https://github.com/gridap/Gridap.jl). It models the coupled physics of Navier-Stokes flow and solute transport (Advection-Diffusion) inside a cylindrical domain.

## Physics Involved
The Boussinesq approximation is used to model buoyancy-driven flows where density differences are sufficiently small to be neglected, except where they appear in terms multiplied by the gravitational acceleration ($g$). 

1. **Fluid Flow (Navier-Stokes):** Solves for the velocity vector field $\mathbf{u}$ and scalar pressure $p$. The fluid is treated as incompressible, but the momentum equation includes a buoyancy body force term proportional to the local solutal concentration $c$:
   $$ \mathbf{F}_b = \rho_0 g \beta_c (c - c_{\text{ref}}) \mathbf{e}_g $$
2. **Solutal Transport (Advection-Diffusion):** Solves for the scalar concentration $c$. The concentration is advected by the fluid velocity field $\mathbf{u}$ and diffuses according to the diffusivity coefficient $D$.

## Code Algorithms
- **Finite Elements Formulation:** The solver uses Taylor-Hood elements (`Q2/Q1`) for the Velocity/Pressure fields to satisfy the Ladyzhenskaya-Babuška-Brezzi (inf-sup) condition intrinsincally. The Concentration field uses `Q2` elements to remain compatible with the velocity space representation.
- **Picard Iterations:** Because the system is non-linearly coupled (convective term $\mathbf{u} \cdot \nabla \mathbf{u}$ and concentration-driven buoyancy), the solver employs a Picard (fixed-point) iteration strategy. In each iteration, it conditionally solves the linearized Navier-Stokes (driven by previous $c$) and then the Advection-Diffusion block (driven by the newly found $\mathbf{u}$).
- **SUPG Stabilization:** Streamline-Upwind/Petrov-Galerkin (SUPG) stabilization is applied to both the Navier-Stokes and Advection-Diffusion operators to prevent spurious node-to-node oscillations at higher mesh Péclet/Reynolds numbers.
- **Iterative Solvers:** To circumvent the massive memory overhead of sparse LU direct solvers in 3D, the system relies on customizable Krylov subspace methods like **BiCGStab** using `IterativeSolvers.jl` combined with Preconditioners via `AlgebraicMultigrid.jl` and `IncompleteLU.jl`.

## Configuration Options (`data/case_options.json`)
You can control the physics parameters, boundary scalars, geometry, and linear solver strategies dynamically without touching the Julia scripts by editing the JSON configuration:
```json
{
    "numerical": {
        "solver_NS": {
            "type": "bicgstabl", // Krylov solver metric (iterativesolvers options)
            "precond": "jacobi", // Preconditioner choice: jacobi, ilu, amg, none
            "reltol": 1e-4       // Relative tolerance for linear solver 
        }
    }
}
```

## Known Weaknesses & Opportunities for Improvement
- **Monolithic Block Preconditioning:** The solver handles the Navier-Stokes velocity and pressure DOFs monolithically. Relying purely on BiCGStab with a simple diagonal (Jacobi) preconditioner or Algebraic Multigrid can cause the number of Krylov iterations to scale poorly on extremely fine meshes with >100k DOFs. A better approach (using `GridapSolvers` or `GridapPETSc`) would be to implement an optimized block-triangular Schur-complement preconditioner that formally decouples the velocity and pressure block preconditioners.
- **Picard Convergence:** The current Picard iterations might occasionally oscillate. Adopting a full non-linear damped Newton-Raphson approach or Anderson acceleration inside Gridap would yield tighter, faster nonlinear convergence bounds.

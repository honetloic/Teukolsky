"""
homogeneous.jl

Solvers for the homogeneous Teukolsky equation in Kerr spacetime.
This module provides functions to solve the radial Teukolsky equation
for ingoing (IN) and upgoing (UP) modes.
"""

using DifferentialEquations
using StaticArrays

_DEFAULTSOLVER = AutoVern9(Rosenbrock23(autodiff=false))
_DEFAULTTOLERANCE = 1e-12;

"""
    TeukolskyHS_up(u, param, r)

Right-hand side of the ODE system for the UP (outgoing) solution of the 
homogeneous Teukolsky equation.

This function defines the first-order ODE system for the Teukolsky equation:
du₁/dr = u₂
du₂/dr = F₁(r)u₂ + F₂(r)u₁

where F₁ = -p(r) and F₂ = -q(r) with H = 1 (upgoing modes).

# Arguments
- `u`: State vector [ψ, dψ/dr]
- `param`: Named tuple containing (s, m, a, omega, lambda)
  - `s`: Spin weight
  - `m`: Azimuthal mode number
  - `a`: Kerr spin parameter
  - `omega`: Mode frequency
  - `lambda`: Eigenvalue
- `r`: Radial coordinate (Boyer-Lindquist)

# Returns
- StaticArray containing [dψ/dr, d²ψ/dr²]
"""
function TeukolskyHS_up(u, param, r)
    H = 1
    dH_dr = 0

    _sF1 = -p(r, param.a, param.s, param.omega, param.m, param.lambda, H, dH_dr, df_dr(r,param.a))
    _sF2 = -q(r, param.a, param.s, param.omega, param.m, param.lambda, H, dH_dr, df_dr(r,param.a))

    return SA[u[2], _sF1*u[2] + _sF2*u[1]]
end

"""
    TeukolskyHS_in(u, param, r)

Right-hand side of the ODE system for the IN (ingoing) solution of the 
homogeneous Teukolsky equation.

This function defines the first-order ODE system for the Teukolsky equation:
du₁/dr = u₂
du₂/dr = F₁(r)u₂ + F₂(r)u₁

where F₁ = -p(r) and F₂ = -q(r) with H = -1 (ingoing modes).

# Arguments
- `u`: State vector [ψ, dψ/dr]
- `param`: Named tuple containing (s, m, a, omega, lambda)
  - `s`: Spin weight
  - `m`: Azimuthal mode number
  - `a`: Kerr spin parameter
  - `omega`: Mode frequency
  - `lambda`: Eigenvalue
- `r`: Radial coordinate (Boyer-Lindquist)

# Returns
- StaticArray containing [dψ/dr, d²ψ/dr²]
"""
function TeukolskyHS_in(u, param, r)
    H = -1
    dH_dr = 0

    _sF1 = -p(r, param.a, param.s, param.omega, param.m, param.lambda, H, dH_dr, df_dr(r,param.a))
    _sF2 = -q(r, param.a, param.s, param.omega, param.m, param.lambda, H, dH_dr, df_dr(r,param.a))

    return SA[u[2], _sF1*u[2] + _sF2*u[1]]
end

"""
    solve_psi_up(s, m, a, omega, lambda, rsin, rsout; odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)

Solve for the UP (outgoing) solution of the homogeneous Teukolsky equation.

This function integrates the Teukolsky equation from the outer boundary (at infinity)
inward to the specified inner radius. The initial conditions are set using the 
asymptotic behavior at infinity.

# Arguments
- `s`: Spin weight of the field (-2 for gravitational perturbations)
- `m`: Azimuthal mode number
- `a`: Kerr spin parameter (0 ≤ a < 1 for sub-extremal black holes)
- `omega`: Mode frequency
- `lambda`: Spin-weighted spheroidal eigenvalue
- `rsin`: Inner radial boundary (starting point, typically near the horizon)
- `rsout`: Outer radial boundary (ending point, typically large radius)

# Keyword Arguments
- `odealgo`: ODE solver algorithm (default: AutoVern9(Rosenbrock23(autodiff=false)))
- `reltol`: Relative tolerance for the ODE solver (default: 1e-12)
- `abstol`: Absolute tolerance for the ODE solver (default: 1e-12)

# Returns
- ODE solution object containing ψ_up(r) and dψ_up/dr

# Throws
- `DomainError`: if rsout ≤ rsin

# Example
```julia
s = -2
m = 2
a = 0.9
omega = 0.5
lambda = 4.0
rsin = 2.0
rsout = 100.0
sol_up = solve_psi_up(s, m, a, omega, lambda, rsin, rsout)
```
"""
function solve_psi_up(s, m, a, omega, lambda, rsin, rsout; odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)
    # Sanity check
    if rsin > rsout
        throw(DomainError(rsout, "rsout ($rsout) must be larger than rsin ($rsin)"))
    end
    rp = r_plus(a)
    # Initial conditions at rs = rsout, the outer boundary
    rout, b1n, psi_inf, psi_inf_prime, psi_inf_double_prime = boundary_conditions_infinity(a, s, omega, m, lambda)
    rsspan = (rout, rsin) # Integrate from rsout to rsin *inward*
    param = (s=s, m=m, a=a, omega=omega, lambda=lambda)
    u0 = SA[psi_inf; psi_inf_prime]
    odeprob = ODEProblem(TeukolskyHS_up, u0, rsspan, param)
    odesoln = solve(odeprob, odealgo; reltol=reltol, abstol=abstol)
    near_infinity_solution(r) = (sum(b1n[i+1] / r^i for i in 0:length(b1n)-1), sum(b1n[i+1] * (-i) / r^(i+1) for i in 1:length(a2n)-1), sum(b1n[i+1] * (i) * (i+1) / r^(i+2) for i in 2:length(a2n)-1))
    return (rin=rsin, numerical_solution=odesoln, rout=rsout, near_infinity_solution=near_infinity_solution)
end

"""
    solve_psi_in(s, m, a, omega, lambda, rsin, rsout; odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)

Solve for the IN (ingoing) solution of the homogeneous Teukolsky equation.

This function integrates the Teukolsky equation from the inner boundary (at the horizon)
outward to the specified outer radius. The initial conditions are set using the 
asymptotic behavior at the horizon.

# Arguments
- `s`: Spin weight of the field (-2 for gravitational perturbations)
- `m`: Azimuthal mode number
- `a`: Kerr spin parameter (0 ≤ a < 1 for sub-extremal black holes)
- `omega`: Mode frequency
- `lambda`: Spin-weighted spheroidal eigenvalue
- `rsin`: Inner radial boundary (ending point, typically near the horizon)
- `rsout`: Outer radial boundary (starting point, typically large radius)

# Keyword Arguments
- `odealgo`: ODE solver algorithm (default: AutoVern9(Rosenbrock23(autodiff=false)))
- `reltol`: Relative tolerance for the ODE solver (default: 1e-12)
- `abstol`: Absolute tolerance for the ODE solver (default: 1e-12)

# Returns
- ODE solution object containing ψ_in(r) and dψ_in/dr

# Throws
- `DomainError`: if rsout ≤ rsin

# Example
```julia
s = -2
m = 2
a = 0.9
omega = 0.5
lambda = 4.0
rsin = 2.0
rsout = 100.0
sol_in = solve_psi_in(s, m, a, omega, lambda, rsin, rsout)
```
"""
function solve_psi_in(s, m, a, omega, lambda, rsin, rsout; odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)
    # Sanity check
    if rsin > rsout
        throw(DomainError(rsout, "rsout ($rsout) must be larger than rsin ($rsin)"))
    end

    # Initial conditions at rs = rsout, the outer boundary
    rp = r_plus(a)
    rin, a2n, psi_hor, psi_hor_prime, psi_hor_double_prime = boundary_conditions_horizon(a, s, omega, m, lambda)
    rsspan = (rin, rsout) # Integrate from rin to rsout *outward*
    param = (s=s, m=m, a=a, omega=omega, lambda=lambda)
    u0 = SA[psi_hor; psi_hor_prime]

    odeprob = ODEProblem(TeukolskyHS_in, u0, rsspan, param)
    odesoln = solve(odeprob, odealgo; reltol=reltol, abstol=abstol)
    near_horizon_solution(r) = (sum(a2n[i+1] * (r - rp)^i for i in 0:length(a2n)-1), sum(a2n[i+1] * i * (r - rp)^(i-1) for i in 1:length(a2n)-1), sum(a2n[i+1] * i * (i-1) * (r - rp)^(i-2) for i in 2:length(a2n)-1))

    return (near_horizon_solution=near_horizon_solution, rin=rin, numerical_solution=odesoln, rout=rsout)
end

"""
    psi_in(s, m, a, omega, lambda, rsin, rsout; odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)
Construct the IN (ingoing) homogeneous Teukolsky solution function.
This function returns a callable that provides the value of the ingoing
homogeneous solution and its first two derivatives at any radius r.

# Arguments
- `s`: Spin weight of the field (-2 for gravitational perturbations)
- `m`: Azimuthal mode number
- `a`: Kerr spin parameter (0 ≤ a < 1 for sub-extremal black holes)
- `omega`: Mode frequency
- `lambda`: Spin-weighted spheroidal eigenvalue
- `rsin`: Inner radial boundary (ending point, typically near the horizon)
- `rsout`: Outer radial boundary (starting point, typically large radius)
# Keyword Arguments
- `odealgo`: ODE solver algorithm (default: AutoVern9(Rosenbrock23(autodiff=false)))
- `reltol`: Relative tolerance for the ODE solver (default: 1e-12)
- `abstol`: Absolute tolerance for the ODE solver (default: 1e-12)
# Returns
- A callable function that takes a radius `r` and returns a tuple
  `(ψ_in(r), dψ_in/dr, d²ψ_in/dr²)`.
"""

function psi_in(s, m, a, omega, lambda, rsin, rsout; odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)
    param = (s=s, m=m, a=a, omega=omega, lambda=lambda)
    psi_in_sols = solve_psi_in(s, m, a, omega, lambda, rsin, rsout)

    psi_in(r) = (r <= psi_in_sols.rin ? psi_in_sols.near_horizon_solution(r)[1] : psi_in_sols.numerical_solution(r)[1]) 
    dpsi_in(r) = (r <= psi_in_sols.rin ? psi_in_sols.near_horizon_solution(r)[2] : psi_in_sols.numerical_solution(r)[2])
    d2psi_in(r) = (r <= psi_in_sols.rin ? psi_in_sols.near_horizon_solution(r)[3] : TeukolskyHS_in((psi_in(r), dpsi_in(r)), param, r)[2])

    return r -> (psi_in(r), dpsi_in(r), d2psi_in(r))
end

function psi_up(s, m, a, omega, lambda, rsin, rsout; odealgo=_DEFAULTSOLVER, reltol=_DEFAULTTOLERANCE, abstol=_DEFAULTTOLERANCE)
    param = (s=s, m=m, a=a, omega=omega, lambda=lambda)
    psi_up_sols = solve_psi_up(s, m, a, omega, lambda, rsin, rsout)

    psi_up(r) = (r <= psi_up_sols.rout ? psi_up_sols.numerical_solution(r)[1] : psi_up_sols.near_infinity_solution(r)[1]) 
    dpsi_up(r) = (r <= psi_up_sols.rout ? psi_up_sols.numerical_solution(r)[2] : psi_up_sols.near_infinity_solution(r)[2])
    d2psi_up(r) = (r <= psi_up_sols.rout ? TeukolskyHS_up((psi_up(r), dpsi_up(r)), param, r)[2] : psi_up_sols.near_infinity_solution(r)[3])

    return r -> (psi_up(r), dpsi_up(r), d2psi_up(r))
end
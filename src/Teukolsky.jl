"""
Teukolsky.jl

FIXME: Add module documentation here.
"""
module Teukolsky

using ForwardDiff

# Include submodules
include("kerr_geometry.jl")
include("boundary_conditions.jl")
include("homogeneous.jl")

# Export geometry functions
export chop_noise, Delta, rho, rho_cc, K, Sigma, f, Gtilde, Utilde, df_dr, r_plus, r_minus

# Export boundary condition functions
export p, q, dq_horizon, dq_infinity, dp_horizon, dp_infinity, c_horizon, A2n, boundary_conditions_horizon, boundary_conditions_infinity, B1n

# Export homogeneous solution functions
export TeukolskyHS_up, TeukolskyHS_in, solve_psi_up, solve_psi_in
export psi_in


end # module

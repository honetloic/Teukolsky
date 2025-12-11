"""
kerr_geometry.jl

Geometric quantities and helper functions for Kerr spacetime.
"""

using ForwardDiff

"""
    chop_noise(x, tol=1e-20)

Helper function to set very small values to zero.
Used to avoid numerical issues in square roots and other operations.

# Arguments
- `x`: Value to chop
- `tol`: Tolerance threshold (default: 1e-20)

# Returns
- Zero if |x| < tol, otherwise x
"""
function chop_noise(x, tol=1e-14)
    return abs(x) < tol ? zero(x) : x
end

"""
    r_plus(a)

Outer event horizon radius r₊ = 1 + √(1 - a²).

# Arguments
- `a`: Kerr spin parameter

# Returns
- Outer horizon radius
"""
function r_plus(a)
    return 1 + sqrt(1 - a^2)
end

"""
    r_minus(a)

Inner event horizon radius r₋ = 1 - √(1 - a²).

# Arguments
- `a`: Kerr spin parameter

# Returns
- Inner horizon radius
"""
function r_minus(a)
    return 1 - sqrt(1 - a^2)
end

"""
    Delta(r, a)

Horizon function Δ(r) = r² - 2r + a².

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter

# Returns
- Value of Δ(r)
"""
function Delta(r, a)
    return r^2 - 2*r + a^2
end

"""
    f(r, a)

dr/drstar

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter

# Returns
- Value of f(r, a)=dr/drstar
"""

function f(r, a)
    return Delta(r, a)/(r^2 + a^2)
end

"""
    rho(r, a, theta)

Complex quantity ρ = (r - ia cos θ)⁻¹.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `theta`: Polar angle

# Returns
- Value of ρ
"""
function rho(r, a, theta)
    return (r - im*a*cos(theta))^(-1)
end

"""
    rho_cc(r, a, theta)

Complex conjugate of ρ: ρ̄ = (r + ia cos θ)⁻¹.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `theta`: Polar angle

# Returns
- Value of ρ̄
"""
function rho_cc(r, a, theta)
    return (r + im*a*cos(theta))^(-1)
end

"""
    K(r, a, omega, m)

K = (r² + a²)ω - ma.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `omega`: Mode frequency
- `m`: Azimuthal mode number

# Returns
- Value of K
"""
function K(r, a, omega, m)
    return (r^2 + a^2)*omega - m*a
end

"""
    Sigma(r, a, theta)

Kerr quantity Σ = r² + a² cos² θ.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `theta`: Polar angle

# Returns
- Value of Σ
"""
function Sigma(r, a, theta)
    return r^2 + a^2*cos(theta)^2
end

"""
    R(r, a, E, lz, C)

Radial potential R(r).

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `E`: Energy per unit mass
- `lz`: Angular momentum per unit mass
- `C`: Carter constant

# Returns
- Value of R(r)
"""
function R(r, a, E, lz, C)
    return ((r^2 + a^2)*E - a*lz)^2 - Delta(r, a)*(r^2 + (lz - a*E)^2 + C)
end

"""
    Tr(r, a, E, lz, C)

Radial contribution to dt/dλ.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `E`: Energy per unit mass
- `lz`: Angular momentum per unit mass
- `C`: Carter constant

# Returns
- Radial contribution to dt/dλ
"""
function Tr(r, a, E, lz, C)
    return (r^2 + a^2)/Delta(r, a)*(E*(r^2 + a^2) - a*lz)
end

"""
    Ttheta(theta, a, E, lz)

Polar contribution to dt/dλ.

# Arguments
- `theta`: Polar angle
- `a`: Kerr spin parameter
- `E`: Energy per unit mass
- `lz`: Angular momentum per unit mass

# Returns
- Polar contribution to dt/dλ
"""
function Ttheta(theta, a, E)
    return -a^2*E*(1 - cos(theta)^2)
end

"""
    Theta(theta, a, E)

Polar potential Θ(θ).

# Arguments
- `theta`: Polar angle
- `a`: Kerr spin parameter
- `E`: Energy per unit mass

# Returns
- Value of Θ(θ)
"""
function Theta(theta, a, E, lz, C)
    return C - (C + a^2*(1 - E^2) + lz^2)*cos(theta)^2 + a^2*(1 - E^2)*cos(theta)^4
end

"""
    dtdlambda(r, theta, a, E, lz, C)

Total time derivative dt/dλ along geodesic.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `theta`: Polar angle
- `a`: Kerr spin parameter
- `E`: Energy per unit mass
- `lz`: Angular momentum per unit mass
- `C`: Carter constant

# Returns
- Value of dt/dλ
"""
function dtdlambda(r, theta, a, E, lz, C)
    return Tr(r, a, E, lz, C) + Ttheta(theta, a, E) + a*lz
end

"""
    drdlambda(r, a, E, lz, C, sign)

Radial velocity dr/dλ along geodesic.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `E`: Energy per unit mass
- `lz`: Angular momentum per unit mass
- `C`: Carter constant
- `sign`: Sign of radial velocity (+1 outgoing, -1 ingoing)

# Returns
- Value of dr/dλ
"""
function drdlambda(r, a, E, lz, C, signr)
    return signr*sqrt(chop_noise(R(r, a, E, lz, C)))
end

"""
    dcosthetadlamba(theta, a, E, lz, C, sign)

Polar velocity d(cos θ)/dλ along geodesic.

# Arguments
- `theta`: Polar angle
- `a`: Kerr spin parameter
- `E`: Energy per unit mass
- `lz`: Angular momentum per unit mass
- `C`: Carter constant
- `sign`: Sign of polar velocity

# Returns
- Value of d(cos θ)/dλ
"""
function dcosthetadlamba(theta, a, E, lz, C, signtheta)
    return signtheta*sqrt(chop_noise(Theta(theta, a, E, lz, C)))
end
#Checked

"""
    Gtilde(r, a, s, M, ω, m, H)

Teukolsky equation coefficient G̃.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `s`: Spin weight
- `M`: Black hole mass (typically M=1)
- `ω`: Mode frequency
- `m`: Azimuthal mode number
- `H`: Height H(r)

# Returns
- Value of G̃
"""
function Gtilde(r, a, s, omega, m, H)
    return a^2 * Delta(r, a) + (r^2 + a^2) * (r * s * (r - 1) - 
           im * r * ((r^2 + a^2) * omega * H + m * a))
end

"""
    Utilde(r, a, s, M, ω, m, λ, H, dH_dr, f)

Teukolsky equation coefficient Ũ.

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `s`: Spin weight
- `M`: Black hole mass (typically M=1)
- `ω`: Mode frequency
- `m`: Azimuthal mode number
- `λ`: Eigenvalue
- `H`: Height H(r)
- `dH_dr`: Derivative of H with respect to r
- `f`: Function f(r,a) = Δ/(r²+a²)

# Returns
- Value of Ũ
"""
function Utilde(r, a, s, omega, m, lambda, H, dH_dr, f)
    return 2 * im * s * omega * r^2 * (r * Delta(r, a) * (1 - H) - 
           (r^2 - a^2) * (1 + H)) - 
           2 * im * a * r * Delta(r, a) * (m + a * omega * H) + 
           Delta(r, a) * (2 * a^2 - r^2 * lambda - 2 * r * (s + 1)) - 
           2 * m * a * omega * r^2 * (r^2 + a^2) * (1 + H) + 
           r^2 * (r^2 + a^2)^2 * (omega^2 * (1 - H^2) + im * omega * f * dH_dr)
end

function df_dr(r,a)
    return ForwardDiff.derivative(r -> f(r, a), r)
end

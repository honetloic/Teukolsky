"""
boundary_conditions.jl

Boundary conditions for the IN and UP solutions of the Teukolsky equation.
"""


"""
    p(r, a, s, omega, m, lambda, H, dH_dr, f)

Teukolsky equation coefficient p.

p = (r² + a²)/Δ * df/dr - 1/(Δ(r² + a²)) * (2G̃)/r

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `s`: Spin weight
- `omega`: Mode frequency
- `m`: Azimuthal mode number
- `lambda`: Eigenvalue
- `H`: Height H(r)
- `dH_dr`: Derivative of H with respect to r
- `f`: Function f(r,a) = Δ/(r²+a²)

# Returns
- Value of p
"""
function p(r, a, s, omega, m, lambda, H, dH_dr, df_dr)
    
    Δ = Delta(r, a)
    r2_a2 = r^2 + a^2
    Gt = Gtilde(r, a, s, omega, m, H)
    
    return (r2_a2 / Δ) * df_dr - (2 * Gt) / (r * Δ * r2_a2)
end

"""
    q(r, a, s, omega, m, lambda, H, dH_dr, f)

Teukolsky equation coefficient q.

q = Ũ / (r² Δ²)

# Arguments
- `r`: Boyer-Lindquist radial coordinate
- `a`: Kerr spin parameter
- `s`: Spin weight
- `omega`: Mode frequency
- `m`: Azimuthal mode number
- `lambda`: Eigenvalue
- `H`: Height H(r)
- `dH_dr`: Derivative of H with respect to r
- `f`: Function f(r,a) = Δ/(r²+a²)

# Returns
- Value of q
"""
function q(r, a, s, omega, m, lambda, H, dH_dr, f)
    Δ = Delta(r, a)
    Ut = Utilde(r, a, s, omega, m, lambda, H, dH_dr, f)
    
    return Ut / (r^2 * Δ^2)
end

function c_horizon(a, omega, m, s)
    rp = r_plus(a)
    rm = r_minus(a)

    return 2*im*(2*rp)/(rp-rm)*(omega - a*m/(2*rp)) + s
end

function dq_horizon(a, s, omega, m, lambda, n)

    typeof_omega = typeof(omega)
    typeof_m = typeof(m)
    rp = r_plus(a)
    rm = r_minus(a)

    if n == 0
        dq = convert(typeof_omega, 0)
    elseif n == 1
        dq = (2*im*a*m + 2*(-1+s)-2*im*a^2*omega + rp * (2+lambda-4*im*rp*s*omega))/((rm-rp)*rp)
    else
        binomial_sum = sum(binomial(convert(typeof_m, n), convert(typeof_m, k)) * (-rm/rp)^(k-1) for k in 2:n)
        dq = 2 * (-1+n) * (-rp)^(-n) + (rm - rp)^(-n) * (2 + lambda - 4*im*rm*s*omega) + 1/rp * 2*n * (rm - rp)^(-n) * ((-1+s)+im*a*(m-a*omega)) * (1 + binomial_sum/n)
    end

    return dq
end

"""
    dq_infinity(a, s, omega, m, lambda, n)

Compute the n-th coefficient dq at infinity for the series expansion.

# Arguments
- `a`: Kerr spin parameter
- `s`: Spin weight
- `omega`: Mode frequency
- `m`: Azimuthal mode number
- `lambda`: Eigenvalue
- `n`: Order of the coefficient

# Returns
- Value of dq_infinity[n]
"""
function dq_infinity(a, s, omega, m, lambda, n)
    rp = r_plus(a)
    rm = r_minus(a)
    
    if n == 0
        return 0
    elseif n == 1
        return 0
    elseif n == 2
        return -(4*a*m*omega + 4*im*s*omega + lambda)
    else  # n > 2
        term1 = (rm - rp)^2 * (2*rm^(-2 + n)*rp - 2*rm*rp^(-2 + n) + 
                2*im*a*m*(-rm^(-2 + n) + rp^(-2 + n)) + 
                2*(-rm^(-2 + n) + rp^(-2 + n))*(1 + s) + 
                (-rm^(-1 + n) + rp^(-1 + n))*lambda)
        
        term2 = 2*im*(rm - rp)^2*(-rm^(-1 + n)*rp + rm*rp^(-1 + n))*omega

        term3 = 4*(a*m*(-rp^n*(2 + n*rm - n*rp) + rm^n*(2 - n*rm + n*rp) + 
                a^2*(rp^(-2 + n)*(rm - n*rm + (-3 + n)*rp) + 
                rm^(-2 + n)*(-(-3 + n)*rm + (-1 + n)*rp))) + 
                im*(-rp^n*(2 + n*rm - n*rp) + rm^n*(2 - n*rm + n*rp) + 
                a^2*(rp^(-2 + n)*((-1 + n)*rm - (-3 + n)*rp) + 
                rm^(-2 + n)*((-3 + n)*rm + rp - n*rp)))*s)*omega
        
        return (term1 + term2 + term3) / (rm - rp)^3
    end
end

function dp_horizon(a, s, omega, m, lambda, n)

    typeof_omega = typeof(omega)
    typeof_m = typeof(m)
    rp = r_plus(a)
    rm = r_minus(a)

    if n == 0
        dp = 1 - c_horizon(a, omega, m, s)
    elseif n == 1
        dp = 1/((rm - rp)^2 * rp) * (-2*rm^2 + a^2*(3 + 2*s + 4*im*rp*omega) + 
             im*rp*(-2*a*m + 2*a^2*omega + im*(rp + 2*s + 2*im*rp^2*omega)))
    else
        dp = 2*(-rp)^(-n) - (rm - rp)^(-n) + (rm - rp)^(-n - 1) * 
             (rm*(2*s) + 2*im*rm^2*omega + 2*im*(-a*m + im*s + a^2*omega))
    end

    return dp
end

"""
    dp_infinity(a, s, omega, m, lambda, n)

Compute the n-th coefficient dp at infinity for the series expansion.

# Arguments
- `a`: Kerr spin parameter
- `s`: Spin weight
- `omega`: Mode frequency
- `m`: Azimuthal mode number
- `lambda`: Eigenvalue
- `n`: Order of the coefficient

# Returns
- Value of dp_infinity[n]
"""
function dp_infinity(a, s, omega, m, lambda, n)
    rp = r_plus(a)
    rm = r_minus(a)
    
    if n == 0
        return 2*im*omega
    elseif n == 1
        return -2*s + 2*im*2*omega
    else  # n > 1
        term1 = rm^(-1 + n) + rp^(-1 + n)
        
        term2 = (2*rm^(-1 + n)*((1 - rm)*s + im*(a*m + (a^2 + rm^2)*omega))) / (rm - rp)
        
        term3 = -(2*rp^(-1 + n)*((1 - rp)*s + im*(a*m + (a^2 + rp^2)*omega))) / (rm - rp)
        
        return term1 + term2 + term3
    end
end

"""
    A2n(n, a, s, omega, m, lambda)

Compute the n-th coefficient A2n for the series expansion near the horizon.

# Arguments
- `n`: Order of the coefficient
- `a`: Kerr spin parameter
- `s`: Spin weight
- `omega`: Mode frequency
- `m`: Azimuthal mode number
- `lambda`: Eigenvalue

# Returns
- Value of A2n[n]
"""
function A2n(a, s, omega, m, lambda, a2n)
    n = length(a2n)
    pq_vec = [j*dp_horizon(a, s, omega, m, lambda, n-j) + dq_horizon(a, s, omega, m, lambda, n-j) for  j in 0:n-1]
    numerator = sum(pq_vec .* a2n)
    denominator = - n * (n - c_horizon(a, omega, m, s))
    return numerator / denominator
end

"""
    B1n(n, a, s, omega, m, lambda)

Compute the n-th coefficient B1n for the series expansion at infinity.

# Arguments
- `n`: Order of the coefficient
- `a`: Kerr spin parameter
- `s`: Spin weight
- `omega`: Mode frequency
- `m`: Azimuthal mode number
- `lambda`: Eigenvalue

# Returns
- Value of B1n[n]
"""
function B1n(a, s, omega, m, lambda, b1n)
    n = length(b1n)
    qp_vec = [dq_infinity(a, s, omega, m, lambda, n-k+1) - k * dp_infinity(a, s, omega, m, lambda, n-k) for  k in 0:n-1]
    numerator = (n-1) / (2*im*omega) * b1n[end] + 1/(2*im*omega*n) * sum(qp_vec .* b1n)
    return numerator
end

"""
    boundary_conditions_horizon(a, s, omega, m, lambda, workingprecision=30, deltarp_init=0.01)

Compute boundary conditions near the horizon for ingoing waves.

# Arguments
- `a`: Kerr spin parameter
- `s`: Spin weight
- `omega`: Mode frequency
- `m`: Azimuthal mode number
- `lambda`: Eigenvalue
- `workingprecision`: Working precision for convergence (default: 30)
- `deltarp_init`: Initial distance from horizon (default: 0.01)

# Returns
- `(cInHor, rin)`: Tuple of coefficients array and starting radius
"""
function boundary_conditions_horizon(a, s, omega, m, lambda)
    
    typeof_omega = typeof(omega)
    workingprecision = eps(typeof_omega)
    rp = r_plus(a)
    rm = r_minus(a)
    deltarp = (rp - rm) / 50
    rin = rp + deltarp
    
    H = -1  # For ingoing waves at horizon
    dH_dr = 0
    fs = f(rin, a)
    dfs = df_dr(rin, a)
    
    phor = p(rin, a, s, omega, m, lambda, H, dH_dr, dfs)
    qhor = q(rin, a, s, omega, m, lambda, H, dH_dr, fs)

    err = Inf
    errold = Inf
    i = 1
    psi_hor = convert(typeof_omega, 1)
    psi_hor_prime = convert(typeof_omega, 0)
    psi_hor_double_prime = convert(typeof_omega, 0)
    
    a2n = [convert(Complex{typeof_omega}, 1)]  # a2n[0] = 1, stored at index 1
    errtab = [1.0]

    while err > 10.0*workingprecision
        # Apparently, it works better when comparing to the MST method to add more terms
        # instead of decreasing the starting radius. Not sure why.
        if mod(i, 30) == 0
            deltarp = deltarp / 2
            rin = rp + deltarp
            phor = p(rin, a, s, omega, m, lambda, H, dH_dr, dfs)
            qhor = q(rin, a, s, omega, m, lambda, H, dH_dr, fs)
        end
        
        # Compute next coefficient using A2n
        new_coeff = A2n(a, s, omega, m, lambda, a2n)
        push!(a2n, new_coeff)
        
        # Build ψ_hor: ψ_hor(r) = 1 + Σ a2n[k] (r - rp)^k
        # a2n[1] corresponds to coefficient for (r-rp)^0, a2n[2] for (r-rp)^1, etc.
        psi_hor += a2n[i+1] * (rin - rp)^i
        psi_hor_prime += a2n[i+1] * i * (rin - rp)^(i-1)
        psi_hor_double_prime += a2n[i+1] * i * (i-1) * (rin - rp)^(i-2)

        # Compute error: |ψ''(rin) + phor * ψ'(rin) + qhor * ψ(rin)|
        err = abs(psi_hor_double_prime + phor * psi_hor_prime + qhor * psi_hor)
        if errold <= err && errold > 10.0*workingprecision && i > 5
            break
        else
            errold = err
        end
        push!(errtab, err)
        i += 1
        
        # Safeguard to avoid runaway computation
        if i > 100
            break
        end
    end
    
    return (rin, a2n, psi_hor, psi_hor_prime, psi_hor_double_prime)
end

function boundary_conditions_infinity(a, s, omega, m, lambda)
    
    typeof_omega = typeof(omega)
    workingprecision = eps(typeof_omega)
    rp = r_plus(a)
    rout = 10*oftype(omega, pi)*(1/abs(omega)+abs(omega)/(1+abs(omega)))
    
    H = 1  # For outgoing waves at infinity
    dH_dr = 0
    fs = f(rout, a)
    dfs = df_dr(rout, a)

    pinf = p(rout, a, s, omega, m, lambda, H, dH_dr, dfs)
    qinf = q(rout, a, s, omega, m, lambda, H, dH_dr, fs)

    err = Inf
    i = 1
    psi_inf = convert(typeof_omega, 1)
    psi_inf_prime = convert(typeof_omega, 0)
    psi_inf_double_prime = convert(typeof_omega, 0)

    b1n = [convert(Complex{typeof_omega}, 1)]  # b1n[0] = 1, stored at index 1
    errtab = [1.0]

    while err > 10.0*workingprecision
        
        # Compute next coefficient using A2n
        new_coeff = B1n(a, s, omega, m, lambda, b1n)
        push!(b1n, new_coeff)

        # Build ψ_inf: ψ_inf(r) = 1 + Σ b1n[k] (r - rp)^k
        # b1n[1] corresponds to coefficient for (r-rp)^0, b1n[2] for (r-rp)^1, etc.
        psi_inf += b1n[i+1] * (rout)^(-i)
        psi_inf_prime += b1n[i+1] * (-i) * (rout)^(-i-1)
        psi_inf_double_prime += b1n[i+1] * (-i) * (-i-1) * (rout)^(-i-2)

        # Compute error: |ψ''(rin) + phor * ψ'(rin) + qhor * ψ(rin)|
        err = abs(psi_inf_double_prime + pinf * psi_inf_prime + qinf * psi_inf)
        
        i += 1
        push!(errtab, err)
        
        # Safeguard to avoid runaway computation
        if i > 100
            break
        end
    end

    return (rout, b1n, psi_inf, psi_inf_prime, psi_inf_double_prime)
end
# Teukolsky–Starobinsky identity, specialized to s=+2 -> s=-2.
# Adapted from Gabriel Andres Piovano's TsolverUpFromTSidentity (MIT);
# see THIRD_PARTY_NOTICES.md. Coefficients are reduced from the supplied
# Mathematica R and R' identities, with common subexpressions shared.
# They map (Rplus,Rplus') to (Rminus,Rminus') after division by 16omega^4.
function _up_ts_physical_coefficients(r, a, m, omega, lp)
    d = Delta(r,a)
    k = (r^2+a^2)*omega-a*m
    t0 = 32im
    t1 = k * t0
    t2 = k ^ 3
    t3 = t0 * t2
    t4 = im * k
    t5 = 96 * t4
    t6 = d * t4
    t7 = 80 * t6
    t8 = r ^ 3
    t9 = im * omega
    t10 = d ^ 2
    t11 = 20 * t10
    t12 = r ^ 2
    t13 = d * t9
    t14 = 64 * t12 * t13
    t15 = 48 * t9
    t16 = t10 * t15
    t17 = 12 * lp
    t18 = t17 * t6
    t19 = d * omega * t0
    t20 = lp * t10
    t21 = 4 * r * t20 * t9
    t22 = k ^ 2
    t23 = r * t22
    t24 = 24 * t13 * t23
    t25 = 8 * t22
    t26 = k ^ 4
    t27 = lp ^ 2
    t28 = 40 * d
    t29 = d * lp
    t30 = omega ^ 2
    t31 = t12 * t30
    t32 = 12 * t10
    t33 = k * omega
    t34 = d * t33
    t35 = 32 * t34
    t36 = r * t35 + t10 * t27 + 24 * t10 + t11 * t33 + t12 * t25 - t12 * t35 + 10 * t20 - t22 * t28 - 16 * t23 - t25 * t29 + t25 + 8 * t26 + t31 * t32
    t37 = 8 * t6
    t38 = im * t2
    t39 = 8 * t38
    t40 = 8 * t9
    t41 = 96 * d
    t42 = 64 * t4
    t43 = lp * t28
    t44 = 160 * t22
    t45 = lp * t4
    t46 = 16 * t45
    t47 = 128 * r
    t48 = 80 * t34
    t49 = 4 * d
    t50 = t27 * t49
    t51 = 32 * lp
    t52 = 16 * r
    t53 = t52 * t9
    t54 = 1 / d
    t55 = 32 * t54
    t56 = t22 * t55
    t57 = t26 * t55
    t58 = 152 * t9
    t59 = t39 * t54
    t60 = 96 * t54
    t61 = 48 * d
    t62 = t12 * t22
    t63 = 16 * t6
    t64 = t45 * t49
    A = -r * t16 + r * t18 + r * t19 - r * t3 - r * t5 + r * t7 - t1 * t8 + t1 + t11 * t9 + t12 * t5 - t14 - t18 + t19 * t8 - t21 + t24 + t3 + t36 - t7
    B = -d ^ 3 * t40 + 16 * im * d * k * r - d * t39 + 4 * im * k * lp * t10 + 24 * im * k * t10 + 8 * im * omega * t10 * t12 - r * t10 * t40 - t12 * t37 - t37
    E = -d * t15 + 8 * im * k ^ 5 * t54 - lp * r * t1 + 36 * lp * t6 + r * t41 + r * t43 - r * t44 + r * t48 + r * t50 + r * t57 + 16 * t12 * t29 * t9 + 256 * t12 * t33 + t12 * t42 + t12 * t46 + t12 * t59 - t13 * t52 + t14 - t16 - t17 * t38 + t19 * t22 - 12 * t20 * t9 + t22 * t51 - t23 * t51 + t23 * t58 + t23 * t60 - t29 * t53 + t30 * t61 * t8 + t31 * t4 * t41 - t31 * t61 - t33 * t47 - 128 * t33 * t8 - t38 * t52 * t54 - 56 * t38 - t4 * t47 + t4 * t50 - t41 + t42 - t43 + t44 + t46 - t48 - t50 + t56 * t8 - t56 - t57 - t58 * t62 + t59 - t60 * t62 + t7
    F = r * t63 + r * t64 + t10 * t53 + t21 - t24 + t32 * t9 + t36 - t63 - t64
    return A, B, E, F
end

# Cancel the identical oscillatory factors analytically. For either spin,
# R_s=Delta^(-s)/r * exp(i omega rstar + i m azimuth) * psi_s.
# The logarithmic derivatives include +a*m, as in the HSC ODE and Mathematica
# resfac; this is independent of PlungeKerr's existing radial wrapper convention.
function _up_ts_state(state, r, a, m, omega, lambda_plus)
    A,B,E,F = _up_ts_physical_coefficients(r,a,m,omega,lambda_plus)
    delta = Delta(r,a)
    phase_prime = ((r^2+a^2)*omega+a*m)/delta
    Lplus = -4*(r-1)/delta - 1/r + im*phase_prime
    Lminus = 4*(r-1)/delta - 1/r + im*phase_prime
    u,v = state[1],state[2]
    vphysical = v+Lplus*u
    scale = inv(16*omega^4*delta^4)
    value = scale*(A*u+B*vphysical)
    derivative = scale*(E*u+F*vphysical)-Lminus*value
    return SA[value,derivative]
end

struct _TransformedUpSolution{S,P}
    positive_solution::S
    parameters::P
end
function (solution::_TransformedUpSolution)(r::Real)
    p=solution.parameters
    return _up_ts_state(solution.positive_solution(r),r,p.a,p.m,p.omega,p.lambda)
end

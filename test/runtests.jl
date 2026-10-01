using Test, Teukolsky, ForwardDiff

@testset "Starobinsky rescaled differential identity" begin
    # Arbitrary local +2 states: test the map independently of any radial solve.
    for a in (0.,.4,-.4), w in (.1,-.2), r in (r_plus(a)+.05,4.,10.,100.)
        lp=1.3; u=.7-.2im; v=-.3+.4im
        positive=(s=2,m=2,a=a,omega=w,lambda=lp)
        negative=(s=-2,m=2,a=a,omega=w,lambda=lp+4)
        dd=TeukolskyHS_up((u,v),positive,r)[2]
        state(x)=Teukolsky._up_ts_state((u+v*(x-r)+dd*(x-r)^2/2,v+dd*(x-r)),x,a,2,w,lp)
        z=state(r)
        derivative(j)=ForwardDiff.derivative(x->real(state(x)[j]),r)+
                      im*ForwardDiff.derivative(x->imag(state(x)[j]),r)
        @test derivative(1) ≈ z[2] rtol=2e-9 atol=1e-10
        @test derivative(2) ≈ TeukolskyHS_up(z,negative,r)[2] rtol=2e-9 atol=1e-10
    end
end

@testset "UP API, branches and direct regression" begin
    for (a,w,lam) in ((0.,.1,4.),(.4,.2,3.6),(-.4,-.2,3.6))
        rp=r_plus(a);rin=rp+1e-4
        st=solve_psi_up(-2,2,a,w,lam,rin,8.)
        @test st.method===:starobinsky
        @test st.spin_integrated==2 && st.lambda_integrated==lam-4
        @test st.integration_solution.prob.p.s==2
        @test last(st.integration_solution.t)==rin
        @test st.rout==first(st.integration_solution.t)
        @test st.requested_rout==8.
        direct=psi_up(-2,2,a,w,lam,rp+.1,8.;method=:direct)
        for r in (rp+.1,3.,5.,8.)
            value=st.numerical_solution(r);reference=direct(r)
            @test value[1] ≈ reference[1] rtol=3e-7
            @test value[2] ≈ reference[2] rtol=3e-7
        end
        for r in (rin,st.rout,1.2st.rout)
            state=r<=st.rout ? st.numerical_solution(r) : st.near_infinity_solution(r)
            @test all(isfinite,state)
        end
        # The asymptotic and numerical representations match at the actual boundary.
        @test st.numerical_solution(st.rout)[1] ≈ st.near_infinity_solution(st.rout)[1] rtol=1e-12
        @test st.numerical_solution(st.rout)[2] ≈ st.near_infinity_solution(st.rout)[2] rtol=1e-12
    end
    # A supplied solver and tolerances must reach the actual +2 solve.
    st=solve_psi_up(-2,2,0.,.2,4.,2.1,8.;odealgo=Teukolsky.Vern9(),reltol=1e-10,abstol=2e-11)
    @test st.integration_solution.alg isa typeof(Teukolsky.Vern9())
    public=psi_up(-2,2,0.,.2,4.,2.1,8.;odealgo=Teukolsky.Vern9(),reltol=1e-10,abstol=2e-11)
    @test public(5.)[1:2] == Tuple(st.numerical_solution(5.))
    @test length(public(1.2st.rout))==3
    @test_throws DomainError public(2.01)
    @test_throws DomainError solve_psi_up(-2,2,0.,0.,4.,2.1,8.)
    @test_throws ArgumentError solve_psi_up(-2,2,0.,.1,4.,2.1,8.;method=:invalid)
    @test_throws ArgumentError solve_psi_up(2,2,0.,.1,0.,2.1,8.;method=:starobinsky)
    ordinary=solve_psi_up(2,2,0.,.1,0.,2.1,8.)
    @test ordinary.method===:direct
    @test ordinary.spin_integrated==2
end

@testset "Independent high-precision Schwarzschild Wronskians" begin
    # User's Mathematica references, 2026-09-30, matching the established IN/UP normalization.
    for (w,reference) in ((.1,-50253.6567371149546695+1304.4800640425402im),
        (.1360825803824856,-12164.642611729414+1833.4516061886084im),
        (.2,-1664.7100094221168+858.5519438646085im),
        (.3,-44.95116629085923+202.602967915062im),
        (.5,30.069530115741246-41.183580124249824im))
        pin=psi_in(-2,2,0.,w,4.,2.1,8.)
        pup=psi_up(-2,2,0.,w,4.,2.1,8.)
        values=ComplexF64[]
        for r in (2.1,3.,5.05,8.)
            d=Delta(r,0.);i=pin(r);u=pup(r)
            # Physical W after cancelling opposite Schwarzschild phase factors.
            W=2d^3/r^2*(i[1]*u[2]-i[2]*u[1]+2im*w*r^2/d*i[1]*u[1])
            push!(values,W)
            @test W ≈ reference rtol=1e-10
        end
        @test maximum(abs.(values./values[3].-1)) < 1e-10
    end
end

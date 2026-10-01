import sympy as S
from sympy.parsing.mathematica import parse_mathematica
from pathlib import Path
import sys
# Usage: python scripts/generate_starobinsky.py /path/to/HSCSolverHomogeneousRadialTeukolsky.wl
source=Path(sys.argv[1]).read_text()
section=source.split('TsolverUpFromTSidentity[s_',1)[1]
lines=[line for line in section.splitlines() if line.strip().startswith(('Rup=Function[{r},','dRup=Function[{r},'))][:2]
assert len(lines)==2
r,a,m,w,lam=S.symbols('r a m omega lambda',real=True)
u,v,d,k=S.symbols('u v d k')
expr=[]
for line in lines:
 text=line.split('Function[{r},',1)[1].removesuffix('];')
 for old,new in [('cprimelm\\[Omega]','cnorm'),('\\[Lambda]op','lam'),('\\[Omega]','w'),('dRupsop[r]','v'),('Rupsop[r]','u'),('\\[CapitalDelta][r]','d'),('KK[r]','k')]:text=text.replace(old,new)
 e=parse_mathematica(text).subs({'M':1,'sop':2,'cnorm':1})
 # substitute by name to unify symbols
 e=e.subs({x:dict(r=r,w=w,lam=lam,u=u,v=v,d=d,k=k).get(str(x),x) for x in e.free_symbols})
 expr.append(S.expand(e))
A,B=expr[0].coeff(u),expr[0].coeff(v)
E,F=expr[1].coeff(u),expr[1].coeff(v)

# Emit the exact s=2 coefficients of the two supplied Mathematica identities.
replacements,outputs=S.cse([A,B,E,F],symbols=S.numbered_symbols("t"))
def code(e):
 return S.julia_code(e).replace(".*","*").replace("./","/").replace(".^","^").replace("lambda","lp")
lines=["    "+str(sym)+" = "+code(e) for sym,e in replacements]
lines += ["    "+name+" = "+code(e) for name,e in zip(("A","B","E","F"),outputs)]
print('\n'.join(lines))
# Independent check that the supplied R' identity differentiates the supplied R
# identity after using the physical s=+2 radial equation.
der=lambda e:S.diff(e,r)+2*(r-1)*S.diff(e,d)+2*r*w*S.diff(e,k)
V=(lam-8*S.I*w*r)/d-k*k/d**2+4*S.I*(r-1)*k/d**2
for expression in (E-der(A)-B*V,F-A-der(B)+6*(r-1)*B/d):
    expression=S.cancel(S.expand(expression).subs({d:r*r-2*r+a*a,k:(r*r+a*a)*w-a*m}))
    assert expression==0
print("# Symbolic R/Rprime consistency checks passed.")

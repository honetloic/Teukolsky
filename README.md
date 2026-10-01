# Teukolsky.jl

**Alpha v0.1.0-alpha.1 · Julia 1.11+ · MIT**

Homogeneous radial Teukolsky solutions in the rescaled psi representation.
For s=-2, the default UP route integrates s=+2 and applies the
Teukolsky–Starobinsky transformation. A direct route remains available.

After the alpha tag is published:

```julia
using Pkg
Pkg.add(url="https://github.com/honetloic/Teukolsky.git", rev="v0.1.0-alpha.1")
using Teukolsky
psi = psi_up(-2,2,0.0,0.1,4.0,2.1,8.0;reltol=1e-12,abstol=1e-12)
psi(4.0) # (psi, dpsi/dr, d²psi/dr²)
```

Both psi_in and psi_up forward solver controls. See docs/src/api/homogeneous.md
for the transformation and domain conventions. Boundary-series accuracy is distinct
from ODE tolerance; static modes and extreme near-horizon limits are restricted.
The high-precision Wronskian fixtures and differential-identity checks are in test/.

Run tests in the installed environment with Pkg.test("Teukolsky"), or run
`julia scripts/ci.jl` from a checkout to test in a fresh environment. Root manifests
are local development files and are not distributed. CI targets Julia 1.11 on
Linux, macOS and Windows; inspect actual CI results before release.

See LICENSE and THIRD_PARTY_NOTICES.md for the MIT terms and upstream attribution.
No GitHub tag is created by these preparation files.

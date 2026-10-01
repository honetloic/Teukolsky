# Changelog

## 0.1.0-alpha.1 (prepared; tag not published by this change)

First portable alpha release, installed from GitHub at `v0.1.0-alpha.1`.
Requires Julia 1.11 or newer; CI targets Julia 1.11 on Linux, macOS and Windows.
MIT licensed. Package identity/UUID is retained.

Stable positive-spin integration and Teukolsky–Starobinsky conversion for s=-2
UP, direct-route fallback, IN/UP solver-control forwarding, corrected exterior
callbacks, and radial regression tests. The upstream MIT notice is retained in
THIRD_PARTY_NOTICES.md. Static and extreme near-horizon limits remain restricted.

Uses OrdinaryDiffEq directly; the unused stochastic/boundary-value solver umbrella
is excluded from the runtime dependency graph.

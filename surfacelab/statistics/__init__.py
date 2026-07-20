"""
surfacelab.statistics — quantifying whether cross-asset coupling is *exploitable*.

A small, self-contained set of snippets that turn the qualitative observation "the ATM
increments are correlated but cross-asset models never help" into empirical numbers:
the cross-asset correlation rho, the own-observation noise r, and the closed-form
exploitability ceiling C = rho*r/((1-rho)+r) that bounds what ANY method could gain.

Entry point:  python3 -m surfacelab.statistics.run
See README.md for the theory and how each file maps to it.
"""

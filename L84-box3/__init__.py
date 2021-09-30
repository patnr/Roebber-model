"""Lorenz-84 coupled with Birchfield's 3-box representation of north Atlantic ocean.

Details taken from `bib.tardif2014coupled`.
Model first proposed by `bib.roebber1995climate`.
"""

# Note that Roebber non-dimensionalises the equations,
# while Tardif uses the original "unit-full" equations.

from collections import namedtuple
import numpy as np
import mpl_tools.place as place

import dapper.mods as modelling

# Lorenz-84 constants
a  = 0.25
b  = 4.0

# Ocean box volumes (m^3)
V0 = 10**16
V1 = 0.832 * V0
V2 = 2.592 * V0
V3 = 10.3  * V0

# Coefficients (m^3 / s)
KT = 10.5 * 10**6  # heat exchange
KZ = 1.0  * 10**6  # vertical eddy diffusion (upper-deep ocean)

# Coefficients of meriodional zonal diabatic heating
F0, F1, F2 = 6.65, 2.0, 47.9
G0, G1, G2 = -3.6, 1.24, 3.81

# Linear fit/parametrisation of equivalent salt flux by eddy energy (m^3 / s)
c1 = 1.25  * 10**6
c2 = 0.156 * 10**6

# Thermal wind balance parameterisation
gamma = 0.06848
# T0    = 278.15 # 5°C. Used by Tardif
T0    = 298.15   # 25°C. Used by Roebber
TA2   = 298.15   # 25°C. Used by both

# MOC constants
# Coefficients for thermal and haline expansion of seawater
alpha = 1.0 * 10**-4   # K^-1
beta  = 8.0 * 10**-4   # psu^-1
# Proportionality constant
mu    = 4.0 * 10**10    # m^3 / s

# 1 time unit <--> 5 days (according to Lorenz'84)
DAY = 1/5
YEAR = 365*DAY
omega = 2 * np.pi / YEAR


StateVector = namedtuple("StateVector",
                         ["x", "y", "z", "T1", "T2", "T3", "S1", "S2", "S3"])


def MOC(X):
    """Also called thermohaline circulation."""
    X = StateVector(*X)
    return mu * ( alpha*(X.T2 - X.T1) - beta*(X.S2 - X.S1) )


@modelling.ens_compatible
def dxdt(state, t):
    x, y, z, T1, T2, T3, S1, S2, S3 = state
    q = MOC(state)

    # Re-used quantities. Use * coz faster than **.
    y2 = y*y
    z2 = z*z

    F = F0 + F1*np.cos(omega * t) + F2*(T2 - T1)/T0
    G = G0 + G1*np.cos(omega * t) + G2*T1/T0

    dx = - y2 - z2 - a*x + a*F
    dy = x*y - b*x*z - y + G
    dz = b*x*y + x*z - z

    Qs = c1 + c2*(y2 + z2)
    TA1 = TA2 - gamma * x

    dT1 = 1/V1 * (.5*q*(T2 - T3) + KT*(TA1 - T1) - KZ*(T1 - T3))
    dT2 = 1/V2 * (.5*q*(T3 - T1) + KT*(TA2 - T2) - KZ*(T2 - T3))
    dT3 = 1/V3 * (.5*q*(T1 - T2) + KZ*(T1  - T3) + KZ*(T2 - T3))

    dS1 = 1/V1 * (.5*q*(S2 - S3) - KZ*(S1 - S3) - Qs)
    dS2 = 1/V2 * (.5*q*(S3 - S1) - KZ*(S2 - S3) + Qs)
    dS3 = 1/V3 * (.5*q*(S1 - S2) + KZ*(S1 - S3) + KZ*(S2 - S3))

    return np.array([dx, dy, dz, dT1, dT2, dT3, dS1, dS2, dS3])


# dt = 1/30   # (4 hours) as in Lorenz'84 -- with rk4
dt = 3/120  # (3 hours) as in Tardif'14 -- with rk2
step = modelling.with_rk4(dxdt, stages=2)


if __name__ == "__main__":
    nYears = 10
    T  = nYears*YEAR
    K  = round(T/dt)
    tt = np.linspace(0, T, K+1)

    np.random.seed(3)
    x0  = np.concatenate(([0]*3, [T0]*3, [1]*3), dtype=float)
    x0 += np.random.randn(9)
    simulator = modelling.with_recursion(step, prog="Simulating")
    xx = simulator(x0, K, t0=0, dt=dt)

    ## Plot
    from matplotlib import pyplot as plt
    lbls = StateVector._fields + ("MOC", )
    fig, axs = place.freshfig(1, figsize=(9, 6), nrows=len(lbls), sharex=True)
    for variable, label, ax in zip(xx.T, lbls, axs):
        ax.plot(tt/YEAR, variable)
        ax.set_ylabel(label)
        if "T" in label:
            ax.axhline(273.15, c="k", lw=0.5)
            # ax.set_ylim(top=300)
    axs[-1].plot(tt/YEAR, MOC(xx.T).T)
    axs[-1].set_ylabel("MOC")
    axs[-1].set_xlabel("Years")
    for ax in axs:
        ax.yaxis.set_label_coords(-.08, .5)
    plt.show()

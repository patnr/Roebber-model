"""Lorenz-84 coupled with Birchfield's 3-box representation of north Atlantic ocean.

Details taken from `bib.tardif2014coupled`.
Model first proposed by `bib.roebber1995climate`.
"""

# Note that Roebber non-dimensionalises the equations,
# while Tardif uses the original "unit-full" equations.

from collections import namedtuple
from pathlib import Path
from dapper.tools.progressbar import progbar
import numpy as np

import dapper.mods as modelling

# Lorenz-84 constants
a  = 0.25
b  = 4.0

# Ocean box volumes
unitV = 10**11  # NB
# unitV = 10**16  # (m^3)
V1 = 0.832 * unitV
V2 = 2.592 * unitV
V3 = 10.3  * unitV

# Coefficients
unitK = 10**6  # (m^3 / s)
KT = 10.5 * unitK  # heat exchange
KZ = 1.0  * unitK  # vertical eddy diffusion (upper-deep ocean)

# Coefficients of meriodional zonal diabatic heating
F0, F1, F2 = 6.65, 2.0, 47.9
G0, G1, G2 = -3.6, 1.24, 3.81

# Linear fit/parametrisation of equivalent salt flux by eddy energy
unitC = 10**6  # (m^3 / s)
c1 = 1.25  * unitC
c2 = 0.156 * unitC

# Thermal wind balance parameterisation
gamma = 0.06848  # Used by Tardif
# T0    = 278.15  # 5°C. Used by Tardif
T0    = 298.15   # 25°C. Used by Roebber
TA2   = 298.15   # 25°C. Used by both

# MOC constants
# Coefficients for thermal and haline expansion of seawater
alpha = 1.0 * 10**-4   # K^-1
beta  = 8.0 * 10**-4   # psu^-1
# Proportionality constant
mu    = 4.0 * 10**10  # m^3 / s

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
    # Constants used in uncoupled L84
    # F = 8.0
    # G = 1.23

    dx = - y2 - z2 - a*x + a*F
    dy = x*y - b*x*z - y + G
    dz = b*x*y + x*z - z
    # Make atoms. constant
    # dx = dy = dz = 0

    Qs = c1 + c2*(y2 + z2)
    TA1 = TA2 - gamma * x

    dT1 = 1/V1 * (.5*q*(T2 - T3) + KT*(TA1 - T1) - KZ*(T1 - T3))
    dT2 = 1/V2 * (.5*q*(T3 - T1) + KT*(TA2 - T2) - KZ*(T2 - T3))
    dT3 = 1/V3 * (.5*q*(T1 - T2) + KZ*(T1  - T3) + KZ*(T2 - T3))

    dS1 = 1/V1 * (.5*q*(S2 - S3) - KZ*(S1 - S3) - Qs)
    dS2 = 1/V2 * (.5*q*(S3 - S1) - KZ*(S2 - S3) + Qs)
    dS3 = 1/V3 * (.5*q*(S1 - S2) + KZ*(S1 - S3) + KZ*(S2 - S3))

    return np.array([dx, dy, dz, dT1, dT2, dT3, dS1, dS2, dS3])


# 1 time unit <--> 5 days (according to Lorenz'84)
DAY = 1/5
YEAR = 365*DAY
omega = 2 * np.pi / YEAR

# If you turn off the atmosphere, then you can use much larger dt
# dt = 1/30   # (4 hours) as in Lorenz'84 -- with rk4
dt = 3/120  # (3 hours) as in Tardif'14 -- with rk2
step = modelling.with_rk4(dxdt, stages=2)


if __name__ == "__main__":
    nYears = 1000
    T  = nYears*YEAR
    K  = round(T/dt)
    tt = np.linspace(0, T, K+1)

    np.random.seed(3)
    xyz0 = np.array([0.97, 0.12, 0.32])
    sal0 = np.array([.2, .9, 1.3])
    tmp0 = np.array([-5, 5, 10]) + T0
    # xyz0 = 0.97, 0.12, 0.32  # L84 mean
    # sal0 = .2, 0.9, 1.3
    # tmp0 = 295, 305, 310
    x0   = (*xyz0, *tmp0, *sal0)
    x0   = x0 + 0*np.random.randn(9)

    def simulator(x, K, t0, dt):
        xx = np.zeros((K+1, len(x)))
        xx[0] = x
        for k in progbar(range(K)):
            tk = t0 + k*dt
            xx[k+1] = step(xx[k], tk, dt)
        return xx
    xx = simulator(x0, K, t0=0, dt=dt)

    ## Plot
    # Group state components
    def groupby(k0):
        states = StateVector(*xx.T)._asdict().items()
        return {"box" + lbl[1:]: v for lbl, v in states if lbl[0] == k0}
    first_letters = list({lbl[0]: None for lbl in StateVector._fields})
    grouped = {k0: groupby(k0) for k0 in first_letters}
    grouped["MOC (1e6)"] = {None: MOC(xx.T) / 1e6}

    from matplotlib import pyplot as plt
    import mpl_tools.place as place
    plt.ion()
    fig, axs = place.freshfig("Time series", figsize=(9, 6),
                              nrows=len(grouped), sharex=True,
                              gridspec_kw={'hspace': 0.0})

    for lbl0, ax in zip(grouped, axs):
        group = grouped[lbl0]
        legion = len(group) > 1
        for i, (lbl1, series) in enumerate(group.items()):
            # ax.plot((tt/YEAR), series,
            ax.plot((tt/YEAR)[::10], series[::10],
                    label=lbl1, c=f"C{i+legion}",
                    lw=(.4 if lbl0 in "xyz" else 2))
        # if lbl0 == "T":
        #     ax.axhline(273.15, c="k", lw=0.5)
        if legion:
            ax.legend(loc="upper right")
        ax.set_ylabel(lbl0)

    axs[-1].set_xlabel("Years")
    # Align ylabels
    for ax in axs:
        ax.yaxis.set_label_coords(-.08, .5)
    fig.tight_layout()

    filename = "time_series"
    i = 0
    while Path(f"{filename}_{i}.png").exists():
        i += 1
    fig.savefig(f"{filename}_{i}.png")
    fig.savefig(f"{filename}_{i}.pdf")

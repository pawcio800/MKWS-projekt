#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 Symulacja termiczna wtryskiwacza SHOWERHEAD (HTP) - WYCINEK 3D (sektor)
============================================================================

Konfiguracja (ustalona z uzytkownikiem):
  - HTP (High-Test Peroxide) = UTLENIACZ w ukladzie bipropelentowym.
  - Para paliwowa: HTP + ETANOL  ->  T_gas ~ 2500 K.
  - Model 3D: KOMORKA JEDNOSTKOWA (sektor) wokol JEDNEGO otworu.
        Powierzchnia czola na 1 otwor = A_face / n_holes  ->  podzialka
        p = sqrt(A_face / n_holes).  Modelujemy prostopadloscian
        p x p x L z jednym osiowym kanalem (otworem) w srodku.
        Sciany boczne = adiabatyczne (symetria miedzy sasiednimi otworami)
        => warunki okresowe.  To odwzorowuje jeden z n_holes otworow.
  - Cel: rozklad temperatury 3D i TEMPERATURA STANU USTALONEGO (t_max=10 s).

Solver: pelne przewodzenie 3D (x, y, z), schemat JAWNY FTCS, wektoryzacja
        numpy (bez scipy). Otwor = walcowy obszar "void" wyciety z metalu;
        sciana kanalu chlodzona konwekcyjnie przez HTP (Dittus-Boelter).

Bilans ciepla:
  - Czolo (z=0): konwekcja gazu  h_gas*(T_gas-T) + radiacja eps*sig*(Tg^4-T^4)
  - Sciana otworu (kanal): h_p*(T - T_prop)   [chlodzenie HTP]
  - Manifold (z=L): h_back*(T - T_prop)
  - Sciany boczne (x,y = 0,p): adiabatyczne (symetria)

Wyjscie: animacja 3D (GIF) nagrzewania bryly + mapy 2D + raport.
Wymagania: numpy, matplotlib   (bez scipy)
============================================================================
"""

import os
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")  # zapis do plikow (PNG/GIF), bez okien
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

SIGMA = 5.670374419e-8  # stala Stefana-Boltzmanna [W/m^2/K^4]
OUTDIR = os.path.dirname(os.path.abspath(__file__))  # zapis zawsze obok skryptu


# ---------------------------------------------------------------------------
#  PARAMETRY (SI)
# ---------------------------------------------------------------------------
@dataclass
class Material:
    name: str = "316L stainless"
    k: float = 16.0
    rho: float = 8000.0
    cp: float = 500.0

    @property
    def alpha(self):
        return self.k / (self.rho * self.cp)


MAT_SS316L   = Material("316L stainless", k=16.0,  rho=8000.0, cp=500.0)
MAT_ALUMINUM = Material("Aluminum 6061",  k=167.0, rho=2700.0, cp=896.0)
MAT_COPPER   = Material("Copper (C101)",  k=390.0, rho=8960.0, cp=385.0)


@dataclass
class Propellant:
    name: str = "HTP ~90%"
    rho: float = 1390.0
    cp: float = 2730.0
    k: float = 0.50
    mu: float = 1.1e-3
    T_in: float = 293.0

    @property
    def Pr(self):
        return self.cp * self.mu / self.k


@dataclass
class GasSide:
    T_gas: float = 2500.0   # HTP+etanol [K]
    h_gas: float = 2000.0   # [W/m^2/K]
    eps: float = 0.30


@dataclass
class Geometry:
    R: float = 0.030        # promien plyty [m] (do wyliczenia podzialki)
    L: float = 0.012        # grubosc plyty [m]
    n_holes: int = 60
    d_hole: float = 1.5e-3

    @property
    def A_face(self):
        return np.pi * self.R**2

    @property
    def pitch(self):
        # podzialka komorki jednostkowej (bok kwadratu na 1 otwor) [m]
        return np.sqrt(self.A_face / self.n_holes)


@dataclass
class Operation:
    mdot_ox: float = 0.5    # calkowity przeplyw HTP [kg/s]
    h_back: float = 1500.0  # chlodzenie manifoldu [W/m^2/K]
    T0: float = 293.0


@dataclass
class SimConfig:
    Nx: int = 26            # komorki w x (podzialka)
    Ny: int = 26            # komorki w y (podzialka)
    Nz: int = 22            # komorki w z (grubosc)
    t_max: float = 10.0     # czas symulacji [s]
    safety: float = 0.5     # wsp. bezpieczenstwa kroku jawnego
    rate_tol: float = 5e-3  # prog stanu ustalonego [K/s]
    n_frames: int = 70      # klatki animacji 3D


# ---------------------------------------------------------------------------
#  KORELACJA - chlodzenie w otworze
# ---------------------------------------------------------------------------
def hole_heat_transfer(prop, geo, op):
    mdot_hole = op.mdot_ox / geo.n_holes
    A_hole = np.pi * geo.d_hole**2 / 4.0
    v = mdot_hole / (prop.rho * A_hole)
    Re = prop.rho * v * geo.d_hole / prop.mu
    Pr = prop.Pr
    Nu = 4.36 if Re < 2300 else 0.023 * Re**0.8 * Pr**0.4
    h_p = Nu * prop.k / geo.d_hole
    return h_p, Re, Nu, v


# ---------------------------------------------------------------------------
#  SOLVER 3D  (FTCS jawny, wektoryzacja numpy)
# ---------------------------------------------------------------------------
def solve(mat, prop, gas, geo, op, cfg):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    p, L = geo.pitch, geo.L
    dx, dy, dz = p / Nx, p / Ny, L / Nz

    # srodki komorek
    xc = (np.arange(Nx) + 0.5) * dx
    yc = (np.arange(Ny) + 0.5) * dy
    zc = (np.arange(Nz) + 0.5) * dz

    # maska metalu: otwor (walec) w srodku, wzdluz osi z
    cx, cy = p / 2, p / 2
    XX, YY = np.meshgrid(xc, yc, indexing="ij")          # [Nx,Ny]
    in_hole2d = (XX - cx)**2 + (YY - cy)**2 < (geo.d_hole / 2)**2
    solid2d = ~in_hole2d
    solid = np.repeat(solid2d[:, :, None], Nz, axis=2)   # [Nx,Ny,Nz]

    # geometria FVM
    A_x, A_y, A_z = dy * dz, dx * dz, dx * dy
    V = dx * dy * dz
    Cap = mat.rho * mat.cp * V
    Gx, Gy, Gz = mat.k * A_x / dx, mat.k * A_y / dy, mat.k * A_z / dz

    h_p, Re, Nu, v_hole = hole_heat_transfer(prop, geo, op)
    Hx, Hy = h_p * A_x, h_p * A_y          # konwekcja na sciance kanalu
    T_prop = prop.T_in

    # --- jawny krok czasu (stabilnosc FTCS)
    diag = (2 * Gx + 2 * Gy + 2 * Gz
            + gas.h_gas * A_z + op.h_back * A_z
            + 2 * Hx + 2 * Hy
            + 4 * gas.eps * SIGMA * gas.T_gas**3 * A_z) / Cap
    dt = cfg.safety / diag
    n_steps = int(np.ceil(cfg.t_max / dt))
    Tgas4 = gas.T_gas**4

    def step(T):
        Q = np.zeros_like(T)
        # --- przewodzenie + sciany kanalu w kierunku x
        both = solid[:-1] & solid[1:]
        fc = np.where(both, Gx * (T[1:] - T[:-1]), 0.0)
        Q[:-1] += fc; Q[1:] -= fc
        lsv = solid[:-1] & ~solid[1:]
        Q[:-1] += np.where(lsv, Hx * (T_prop - T[:-1]), 0.0)
        rsv = ~solid[:-1] & solid[1:]
        Q[1:] += np.where(rsv, Hx * (T_prop - T[1:]), 0.0)
        # --- kierunek y
        both = solid[:, :-1] & solid[:, 1:]
        fc = np.where(both, Gy * (T[:, 1:] - T[:, :-1]), 0.0)
        Q[:, :-1] += fc; Q[:, 1:] -= fc
        lsv = solid[:, :-1] & ~solid[:, 1:]
        Q[:, :-1] += np.where(lsv, Hy * (T_prop - T[:, :-1]), 0.0)
        rsv = ~solid[:, :-1] & solid[:, 1:]
        Q[:, 1:] += np.where(rsv, Hy * (T_prop - T[:, 1:]), 0.0)
        # --- kierunek z (tylko przewodzenie; otwor jest osiowy)
        bothz = solid[:, :, :-1] & solid[:, :, 1:]
        fcz = np.where(bothz, Gz * (T[:, :, 1:] - T[:, :, :-1]), 0.0)
        Q[:, :, :-1] += fcz; Q[:, :, 1:] -= fcz
        # --- czolo gazu (z=0)
        f0 = solid[:, :, 0]
        q0 = gas.h_gas * A_z * (gas.T_gas - T[:, :, 0]) \
            + gas.eps * SIGMA * A_z * (Tgas4 - T[:, :, 0]**4)
        Q[:, :, 0] += np.where(f0, q0, 0.0)
        # --- manifold (z=L)
        fL = solid[:, :, -1]
        Q[:, :, -1] += np.where(fL, op.h_back * A_z * (T_prop - T[:, :, -1]), 0.0)

        Tn = T + (dt / Cap) * Q
        Tn[~solid] = T_prop
        return Tn

    # --- petla czasowa + zapis klatek
    T = np.where(solid, op.T0, T_prop)
    snap_every = max(1, n_steps // 240)
    snaps, snap_t = [T.copy()], [0.0]
    probe = {"max": [], "czolo-metal (max)": [], "manifold": [], "min": []}
    t_hist = []
    t, steady_time = 0.0, None
    for s in range(n_steps):
        Tn = step(T)
        rate = np.max(np.abs(Tn[solid] - T[solid])) / dt
        T = Tn
        t += dt
        t_hist.append(t)
        probe["max"].append(T[solid].max())
        probe["czolo-metal (max)"].append(T[:, :, 0][solid2d].max())
        probe["manifold"].append(T[:, :, -1][solid2d].mean())
        probe["min"].append(T[solid].min())
        if (s + 1) % snap_every == 0:
            snaps.append(T.copy()); snap_t.append(t)
        if steady_time is None and rate < cfg.rate_tol:
            steady_time = t
        if steady_time is not None and t > steady_time + 3 * dt:
            break
    if snap_t[-1] != t:
        snaps.append(T.copy()); snap_t.append(t)

    # bilans energii -> cieplo odbierane przez HTP (konwekcja kanal + manifold)
    Q_to_prop = _heat_to_prop(T, solid, Hx, Hy, op.h_back, A_z, T_prop)
    dT_prop = Q_to_prop * geo.n_holes / (op.mdot_ox * prop.cp)

    return dict(
        T=T, solid=solid, solid2d=solid2d, in_hole2d=in_hole2d,
        xc=xc, yc=yc, zc=zc, dx=dx, dy=dy, dz=dz, p=p, L=L,
        snaps=np.array(snaps), snap_t=np.array(snap_t),
        t=np.array(t_hist), probe={k: np.array(v) for k, v in probe.items()},
        steady_time=steady_time, dt=dt, n_steps=s + 1,
        h_p=h_p, Re=Re, Nu=Nu, v_hole=v_hole,
        Q_to_prop=Q_to_prop, dT_prop=dT_prop, T_prop=T_prop)


def _heat_to_prop(T, solid, Hx, Hy, h_back, A_z, T_prop):
    """Sumaryczne cieplo [W] odbierane przez HTP w jednej komorce jednostkowej."""
    Q = 0.0
    # sciany kanalu (x)
    lsv = solid[:-1] & ~solid[1:]
    Q += np.sum(np.where(lsv, Hx * (T[:-1] - T_prop), 0.0))
    rsv = ~solid[:-1] & solid[1:]
    Q += np.sum(np.where(rsv, Hx * (T[1:] - T_prop), 0.0))
    # sciany kanalu (y)
    lsv = solid[:, :-1] & ~solid[:, 1:]
    Q += np.sum(np.where(lsv, Hy * (T[:, :-1] - T_prop), 0.0))
    rsv = ~solid[:, :-1] & solid[:, 1:]
    Q += np.sum(np.where(rsv, Hy * (T[:, 1:] - T_prop), 0.0))
    # manifold
    fL = solid[:, :, -1]
    Q += np.sum(np.where(fL, h_back * A_z * (T[:, :, -1] - T_prop), 0.0))
    return float(Q)


# ---------------------------------------------------------------------------
#  RAPORT
# ---------------------------------------------------------------------------
def estimate_tau(t, y):
    y0, yss = y[0], y[-1]
    if abs(yss - y0) < 1e-6:
        return np.nan
    target = y0 + 0.632 * (yss - y0)
    k = np.argmax(y >= target) if yss > y0 else np.argmax(y <= target)
    return t[k] if k > 0 else np.nan


def report(h, mat, prop, gas, geo, op, cfg):
    K2C = lambda T: T - 273.15
    T = h["T"]; solid = h["solid"]; s2 = h["solid2d"]
    print("=" * 70)
    print(" RAPORT 3D - wtryskiwacz showerhead, sektor wokol 1 otworu (HTP+etanol)")
    print("=" * 70)
    print(f" Material   : {mat.name}  (k={mat.k} W/mK, alpha={mat.alpha:.2e} m2/s)")
    print(f" Gaz        : T_gas={K2C(gas.T_gas):.0f} C, h_gas={gas.h_gas} W/m2K, eps={gas.eps}")
    print(f" Komorka    : podzialka p={h['p']*1e3:.2f} mm, L={h['L']*1e3:.1f} mm, "
          f"otwor d={geo.d_hole*1e3:.2f} mm")
    print(f" Siatka     : {cfg.Nx}x{cfg.Ny}x{cfg.Nz}, dt={h['dt']*1e3:.3f} ms, "
          f"krokow={h['n_steps']}")
    print("-" * 70)
    print(" Kanal HTP:")
    print(f"   v={h['v_hole']:.2f} m/s, Re={h['Re']:.0f}, Nu={h['Nu']:.1f}, "
          f"h_p={h['h_p']:.0f} W/m2K")
    print(f"   cieplo/otwor Q={h['Q_to_prop']:.1f} W, przyrost temp HTP "
          f"dT={h['dT_prop']:.2f} K")
    print("-" * 70)
    face = T[:, :, 0][s2]
    print(" Temperatury STANU USTALONEGO:")
    print(f"   czolo metal (max): {K2C(face.max()):8.1f} C  (najgoretszy punkt miedzy otworami)")
    print(f"   czolo metal (min): {K2C(face.min()):8.1f} C  (przy sciance kanalu)")
    print(f"   roznica na czole : {face.max()-face.min():8.1f} K  <-- gradient wokol otworu")
    print(f"   manifold (sred.) : {K2C(T[:, :, -1][s2].mean()):8.1f} C")
    print(f"   MAKS. w bryle    : {K2C(T[solid].max()):8.1f} C")
    print(f"   MIN. w bryle     : {K2C(T[solid].min()):8.1f} C")
    print("-" * 70)
    tau = estimate_tau(h["t"], h["probe"]["max"])
    st = h["steady_time"]
    print(" Dynamika:")
    print(f"   stala czasowa tau: {tau:.2f} s (63% przyrostu)")
    if st:
        print(f"   czas ustalenia   : {st:.2f} s (|dT/dt|<{cfg.rate_tol} K/s)")
    else:
        print(f"   stan ustalony NIE w pelni osiagniety w {cfg.t_max} s")
    print("=" * 70)


# ---------------------------------------------------------------------------
#  WIZUALIZACJA 3D (GIF) + mapy 2D
# ---------------------------------------------------------------------------
def _downsample(solid, T, rf):
    """Blokowy downsampling renderu (rf-krotny). Zwraca (solid_c, T_c)."""
    Nx, Ny, Nz = solid.shape
    nx, ny, nz = Nx // rf, Ny // rf, Nz // rf
    s = solid[:nx * rf, :ny * rf, :nz * rf].reshape(nx, rf, ny, rf, nz, rf)
    t = T[:nx * rf, :ny * rf, :nz * rf].reshape(nx, rf, ny, rf, nz, rf)
    cnt = s.sum((1, 3, 5))
    solid_c = cnt > (rf**3) * 0.4
    T_c = np.where(cnt > 0, (t * s).sum((1, 3, 5)) / np.maximum(cnt, 1), np.nan)
    return solid_c, T_c


def make_gif(h, gas, fname="injector_thermal_3d.gif", fps=12, rf=2, n_frames=40):
    K2C = lambda T: T - 273.15
    out = os.path.join(OUTDIR, fname)

    snaps_all = h["snaps"]; st_all = h["snap_t"]; solid = h["solid"]
    # zageszczenie klatek na poczatku (cala dynamika w pierwszych sekundach)
    nF = min(n_frames, len(snaps_all))
    sel = np.unique(np.round(np.linspace(0, 1, nF) ** 0.5
                             * (len(snaps_all) - 1)).astype(int))
    snaps, st = snaps_all[sel], st_all[sel]

    # downsampling renderu + przekroj (wycieta cwiartka odslania kanal)
    solid_c, _ = _downsample(solid, snaps[0], rf)
    frames_T = [_downsample(solid, s, rf)[1] for s in snaps]
    nxc, nyc, nzc = solid_c.shape
    I, J = np.meshgrid(np.arange(nxc), np.arange(nyc), indexing="ij")
    notch = (I >= nxc // 2)[:, :, None] & (J >= nyc // 2)[:, :, None]
    vis = solid_c & ~np.repeat(notch, nzc, axis=2)

    vmin = K2C(min(np.nanmin(T) for T in frames_T))
    vmax = K2C(max(np.nanmax(T) for T in frames_T))
    cmap = plt.get_cmap("inferno"); norm = Normalize(vmin, vmax)

    vis_r = vis[:, :, ::-1]          # gaz na GORZE (odwrocone z)
    p, L = h["p"], h["L"]

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    axT = fig.add_subplot(1, 2, 2)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10); cb.set_label("T [C]")

    # --- prawy panel: krzywe T(t) (rysowane raz) + ruchomy kursor czasu
    tt = h["t"]
    curves = ["czolo-metal (max)", "min", "manifold"]
    labels = {"czolo-metal (max)": "czolo metal (max)",
              "min": "min (przy kanale)", "manifold": "manifold"}
    cols = {"czolo-metal (max)": "tab:red", "min": "tab:blue",
            "manifold": "tab:green"}
    y_face = K2C(h["probe"]["czolo-metal (max)"])
    for kk in curves:
        axT.plot(tt, K2C(h["probe"][kk]), lw=2, color=cols[kk], label=labels[kk])
    if h["steady_time"]:
        axT.axvline(h["steady_time"], color="gray", ls="--", alpha=0.6,
                    label=f"ustalenie {h['steady_time']:.1f} s")
    cursor = axT.axvline(0.0, color="k", lw=1.3)
    dot, = axT.plot([], [], "o", color="tab:red", ms=8)
    axT.set_xlabel("czas [s]"); axT.set_ylabel("temperatura [C]")
    axT.set_title("Nagrzewanie -> stan ustalony")
    axT.grid(alpha=0.3); axT.legend(loc="center right", fontsize=9)
    axT.set_xlim(0, tt[-1])

    def update(kf):
        ax.clear()
        Tr = K2C(frames_T[kf][:, :, ::-1])
        colors = cmap(norm(np.nan_to_num(Tr, nan=vmin)))
        ax.voxels(vis_r, facecolors=colors, shade=True)   # bez krawedzi = szybko
        ax.set_box_aspect((p, p, L))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (gora=gazy)")
        ax.set_title(f"Sektor 3D (przekroj wokol kanalu)\n"
                     f"t = {st[kf]:5.2f} s    Tmax = {np.nanmax(Tr):.0f} C")
        ax.view_init(elev=24, azim=-45 + 20 * np.sin(kf / len(snaps) * np.pi))
        # kursor + kropka na krzywej czola
        cursor.set_xdata([st[kf], st[kf]])
        j = min(int(np.searchsorted(tt, st[kf])), len(y_face) - 1)
        dot.set_data([st[kf]], [y_face[j]])
        return ()

    anim = animation.FuncAnimation(fig, update, frames=len(snaps),
                                   interval=1000 / fps, blit=False)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF 3D zapisany: {out}  ({len(snaps)} klatek, render {nxc}x{nyc}x{nzc})")


def make_plots(h, save_prefix="injector_thermal"):
    K2C = lambda T: T - 273.15
    pre = os.path.join(OUTDIR, save_prefix)
    T = h["T"]; s2 = h["solid2d"]; p = h["p"]

    # 1) T(t)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for kk, y in h["probe"].items():
        ax1.plot(h["t"], K2C(y), lw=2, label=kk)
    if h["steady_time"]:
        ax1.axvline(h["steady_time"], color="gray", ls="--", alpha=0.7,
                    label=f"ustalenie {h['steady_time']:.1f} s")
    ax1.set_xlabel("czas [s]"); ax1.set_ylabel("temperatura [C]")
    ax1.set_title("Nagrzewanie sektora -> stan ustalony")
    ax1.legend(); ax1.grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig(f"{pre}_transient.png", dpi=130)

    # 2) mapa czola gazu (z=0) - rozklad wokol otworu
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    face = np.where(s2, T[:, :, 0], np.nan)
    im = ax2.imshow(K2C(face).T, origin="lower", cmap="inferno",
                    extent=[0, p * 1e3, 0, p * 1e3])
    cb = fig2.colorbar(im, ax=ax2); cb.set_label("T [C]")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
    ax2.set_title("Czolo (strona gazow) - rozklad T wokol otworu")
    fig2.tight_layout(); fig2.savefig(f"{pre}_face.png", dpi=130)

    # 3) profil przez grubosc: w narozniku (najdalej od otworu) i przy scianie
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    z_mm = h["zc"] * 1e3
    ax3.plot(z_mm, K2C(T[0, 0, :]), "o-", label="naroznik (daleko od otworu)")
    ic = T.shape[0] // 2
    # komorka metalu najblizsza otworowi w linii srodkowej
    col = np.where(s2[ic], np.arange(T.shape[1]), -1)
    jn = col[col >= 0]
    jnear = jn[np.argmin(np.abs(jn - T.shape[1] // 2))]
    ax3.plot(z_mm, K2C(T[ic, jnear, :]), "s-", label="przy sciance kanalu")
    ax3.set_xlabel("z [mm]  (0 = gazy)"); ax3.set_ylabel("temperatura [C]")
    ax3.set_title("Profil temperatury przez grubosc plyty")
    ax3.legend(); ax3.grid(alpha=0.3)
    fig3.tight_layout(); fig3.savefig(f"{pre}_profile.png", dpi=130)

    for f in (fig1, fig2, fig3):
        plt.close(f)
    print(f"Mapy 2D zapisane w: {OUTDIR}/ ({save_prefix}_*.png)")


# ---------------------------------------------------------------------------
def main():
    mat  = MAT_SS316L           # lub MAT_ALUMINUM / MAT_COPPER
    prop = Propellant()
    gas  = GasSide()
    geo  = Geometry()
    op   = Operation()
    cfg  = SimConfig()

    h = solve(mat, prop, gas, geo, op, cfg)
    report(h, mat, prop, gas, geo, op, cfg)
    make_plots(h)
    make_gif(h, gas)


if __name__ == "__main__":
    main()

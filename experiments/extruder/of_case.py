"""OpenFOAM ケース生成（ゲート G3 用）.

展開チャネル断面を blockMesh で切り、フライトを topoSet + subsetMesh で刳り抜く。
x 両端と z 両端は cyclic。x 周期の圧力跳びは体積力 (−β, 0, −G) に還元済みなので
fixedJump は不要、`simpleFoam` のまま回る（設計文書 §2.1.1, §5）。

**格子は ykep-cae と同じ dx/dy を使う。** `ScrewGeometryProcess` の幅配列は
区間ごとに等比数列なので、blockMesh の多区間 grading `((f n r) ...)` で
そのまま再現できる（r = 末尾セル幅 / 先頭セル幅）。これで「格子が違うから
合わない」を切り分けから消せる。一致は compare_openfoam.py がセル中心座標で検査する。

密度 ρ = 1 の運動学的単位で書く（simpleFoam は p/ρ を解く）ので、
nu = μ [m²/s]、体積力は (−β, 0, −G) [m/s²] と数値がそのまま移る。

使い方:
    .venv/bin/python experiments/extruder/of_case.py --model newtonian --out /tmp/of-g3a
    .venv/bin/python experiments/extruder/of_case.py --model powerlaw --K 2e4 --n 0.4 --out /tmp/of-g3b
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np

from xkep_cae_fluid.extruder import (
    NewtonianViscosity,
    PowerLawViscosity,
    ScrewGeometryProcess,
    ScrewSpec,
)

DEFAULT_SPEC = ScrewSpec(D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100.0 / 60.0)
"""設計文書 §6 の仮 40mm 機。格子既定値 (200/48/60/20) → 248×80 = 19,840 セル."""

DEFAULT_G = 5.0e6
"""背圧勾配 [Pa/m]。G4b と build_fields.py に合わせる."""


def foam_header(cls: str, obj: str, location: str | None = None) -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n"
    )


def _f(x: float) -> str:
    return f"{x:.12g}"


def _write(case: str, rel: str, body: str) -> None:
    path = os.path.join(case, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)


@dataclass(frozen=True)
class Section:
    """blockMesh 多区間 grading の 1 区間."""

    widths: np.ndarray

    @property
    def length(self) -> float:
        return float(self.widths.sum())

    @property
    def n(self) -> int:
        return int(self.widths.shape[0])

    @property
    def ratio(self) -> float:
        """末尾セル幅 / 先頭セル幅（blockMesh の expansion ratio の定義）."""
        return float(self.widths[-1] / self.widths[0])


def grading(sections: list[Section]) -> str:
    total = sum(s.length for s in sections)
    parts = " ".join(f"({_f(s.length / total)} {s.n} {_f(s.ratio)})" for s in sections)
    return f"({parts})"


def x_sections(spec: ScrewSpec) -> list[Section]:
    """`ScrewGeometryProcess._build_dx` と同じ分割: 半チャネル / ランド / 半チャネル."""
    dx = ScrewGeometryProcess._build_dx(spec)
    if spec.e <= 0.0:
        return [Section(dx)]
    n_half = max(2, spec.nx_channel // 2)
    n_land = max(1, spec.nx_land)
    return [
        Section(dx[:n_half]),
        Section(dx[n_half : n_half + n_land]),
        Section(dx[n_half + n_land :]),
    ]


def y_sections(spec: ScrewSpec) -> list[Section]:
    """`ScrewGeometryProcess._build_dy` と同じ分割: バルク / 隙間."""
    dy = ScrewGeometryProcess._build_dy(spec)
    if spec.delta <= 0.0 or spec.n_gap <= 0:
        return [Section(dy)]
    return [Section(dy[: spec.ny_bulk]), Section(dy[spec.ny_bulk :])]


def transport_properties(model) -> str:
    """ykep-cae の粘度モデル → transportProperties（ρ = 1 なので nu = μ）.

    OpenFOAM の powerLaw は nu = max(nuMin, min(nuMax, k·γ̇^(n−1)))、
    γ̇ = √2·|symm(∇U)|。これは strain_rate() の γ̇ = √(2 D:D) と同じ定義なので
    k = K、n = n がそのまま対応する（of_powerlaw_check.py が 1D 厳密解で確認）。
    nuMax は ykep 側の γ̇_min クランプと同じ値にして、クランプの掛かり方まで揃える。
    """
    body = foam_header("dictionary", "transportProperties", "constant")
    if isinstance(model, NewtonianViscosity):
        body += f"transportModel  Newtonian;\n\nnu              {_f(model.mu)};\n"
        return body
    if isinstance(model, PowerLawViscosity):
        nu_max = min(model.K * model.gamma_min ** (model.n - 1.0), model.mu_max)
        body += (
            "transportModel  powerLaw;\n\n"
            f"nu              {_f(nu_max)};\n\n"
            "powerLawCoeffs\n{\n"
            f"    nuMax       {_f(nu_max)};\n"
            f"    nuMin       {_f(model.K * 1.0e6 ** (model.n - 1.0))};\n"
            f"    k           {_f(model.K)};\n"
            f"    n           {_f(model.n)};\n"
            "}\n"
        )
        return body
    msg = f"OpenFOAM に対応づけできない粘度モデル: {type(model).__name__}"
    raise TypeError(msg)


def _control_dict(end_time: int, write_interval: int) -> str:
    return foam_header("dictionary", "controlDict", "system") + (
        "application     simpleFoam;\n"
        "startFrom       startTime;\nstartTime       0;\nstopAt          endTime;\n"
        f"endTime         {end_time};\ndeltaT          1;\n"
        f"writeControl    timeStep;\nwriteInterval   {write_interval};\n"
        "purgeWrite      1;\nwriteFormat     ascii;\nwritePrecision  12;\n"
        "writeCompression off;\ntimeFormat      general;\nrunTimeModifiable false;\n"
    )


def _fv_schemes() -> str:
    return foam_header("dictionary", "fvSchemes", "system") + (
        "ddtSchemes      { default steadyState; }\n"
        "gradSchemes     { default Gauss linear; }\n"
        "divSchemes\n{\n    default         none;\n"
        "    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
        "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n"
        "laplacianSchemes { default Gauss linear corrected; }\n"
        "interpolationSchemes { default linear; }\n"
        "snGradSchemes   { default corrected; }\n"
    )


def _fv_solution(
    p_tol: float, u_tol: float, *, relax_u: float = 0.999, residual_control: bool = True
) -> str:
    """SIMPLEC。**U の緩和係数は 0.999 にする（既定の 0.7〜0.9 では収束しない）.**

    緩和係数 α は擬似時間刻み Δτ = α/(1−α)·V/a_P を与える。クリープ流れでは
    a_P ≈ 2ν/Δy² が支配的で、最も滑らかな誤差モード（固有値 ν(π/H)²）の減衰率は
    1/(1 + α/(1−α)·(πΔy/H)²/2) になる。Δy = 5 μm の隙間セルでは α = 0.9 で
    1 反復あたり 1 − 1e-4 しか減らず、数十万反復かかる（1D で実測: α=0.9 で 5000 反復
    後も残差 1e-3、α=0.999 なら 20 反復で 1e-11）。α = 1 は SIMPLEC の 1/(1/rAU − H1)
    が純拡散でゼロ割りになる（実測 sigFpe）ので 0.999 が上限。
    """
    rc = f"    residualControl {{ p {_f(p_tol)}; U {_f(u_tol)}; }}\n" if residual_control else ""
    return foam_header("dictionary", "fvSolution", "system") + (
        "solvers\n{\n"
        "    p\n    {\n        solver          GAMG;\n        smoother        DICGaussSeidel;\n"
        "        tolerance       1e-12;\n        relTol          0.01;\n    }\n"
        "    U\n    {\n        solver          PBiCGStab;\n        preconditioner  DILU;\n"
        "        tolerance       1e-14;\n        relTol          0.01;\n    }\n}\n\n"
        "SIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n    consistent      yes;\n"
        "    pRefCell        0;\n    pRefValue       0;\n"
        f"{rc}}}\n\n"
        f"relaxationFactors\n{{\n    equations {{ U {_f(relax_u)}; }}\n    fields    {{ p 1.0; }}\n}}\n"
    )


def _fv_options(force: tuple[float, float, float]) -> str:
    fx, fy, fz = force
    return foam_header("dictionary", "fvOptions", "system") + (
        "momentumSource\n{\n    type            vectorSemiImplicitSource;\n"
        "    selectionMode   all;\n    volumeMode      specific;\n"
        "    sources\n    {\n"
        f"        U {{ explicit ({_f(fx)} {_f(fy)} {_f(fz)}); implicit 0; }}\n"
        "    }\n}\n"
    )


def _turbulence() -> str:
    return (
        foam_header("dictionary", "turbulenceProperties", "constant") + "simulationType  laminar;\n"
    )


def write_channel_case(
    case: str, spec: ScrewSpec, G: float, model, *, end_time: int = 20000
) -> None:
    """展開チャネル断面のケース一式を書き出す（blockMesh 前）.

    フライトの刳り抜きは topoSetDict（boxToCell → invert）で行い、
    露出した面は subsetMesh の `-patch screw` でスクリュー壁に合流させる。
    """
    W_t, H = spec.W_t, spec.H
    dz = H  # z 1 セルの厚み。cyclic なので値は結果に影響しない
    xs, ys = x_sections(spec), y_sections(spec)
    nx = sum(s.n for s in xs)
    ny = sum(s.n for s in ys)

    verts = [
        (0.0, 0.0, 0.0),
        (W_t, 0.0, 0.0),
        (W_t, H, 0.0),
        (0.0, H, 0.0),
        (0.0, 0.0, dz),
        (W_t, 0.0, dz),
        (W_t, H, dz),
        (0.0, H, dz),
    ]
    vtxt = "\n".join(f"    ({_f(x)} {_f(y)} {_f(z)})" for x, y, z in verts)
    body = foam_header("dictionary", "blockMeshDict", "system") + (
        "scale 1;\n\n"
        f"vertices\n(\n{vtxt}\n);\n\n"
        "blocks\n(\n"
        f"    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1)\n"
        f"    simpleGrading ({grading(xs)} {grading(ys)} 1)\n"
        ");\n\nedges ();\n\n"
        "boundary\n(\n"
        "    screw  { type wall;   faces ((0 1 5 4)); }\n"
        "    barrel { type wall;   faces ((3 7 6 2)); }\n"
        "    left   { type cyclic; neighbourPatch right; faces ((0 4 7 3)); }\n"
        "    right  { type cyclic; neighbourPatch left;  faces ((1 2 6 5)); }\n"
        "    front  { type cyclic; neighbourPatch back;  faces ((0 3 2 1)); }\n"
        "    back   { type cyclic; neighbourPatch front; faces ((4 5 6 7)); }\n"
        ");\n\nmergePatchPairs ();\n"
    )
    _write(case, "system/blockMeshDict", body)

    x_lo, x_hi = 0.5 * (W_t - spec.e), 0.5 * (W_t + spec.e)
    y_top = H - spec.delta
    _write(
        case,
        "system/topoSetDict",
        foam_header("dictionary", "topoSetDict", "system") + "actions\n(\n"
        "    { name c0; type cellSet; action new; source boxToCell;\n"
        f"      box ({_f(x_lo)} -1 -1) ({_f(x_hi)} {_f(y_top)} 1); }}\n"
        "    { name c0; type cellSet; action invert; }\n"
        ");\n",
    )

    _write(case, "system/controlDict", _control_dict(end_time, end_time))
    _write(case, "system/fvSchemes", _fv_schemes())
    _write(case, "system/fvSolution", _fv_solution(1e-8, 1e-9))
    _write(case, "system/fvOptions", _fv_options((-spec.beta(G), 0.0, -G)))
    _write(case, "constant/turbulenceProperties", _turbulence())
    _write(case, "constant/transportProperties", transport_properties(model))

    ub, wb = spec.u_barrel, spec.w_barrel
    _write(
        case,
        "0/U",
        foam_header("volVectorField", "U", "0")
        + "dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n"
        "    screw  { type noSlip; }\n"
        f"    barrel {{ type fixedValue; value uniform ({_f(ub)} 0 {_f(wb)}); }}\n"
        "    left   { type cyclic; }\n    right  { type cyclic; }\n"
        "    front  { type cyclic; }\n    back   { type cyclic; }\n}\n",
    )
    _write(
        case,
        "0/p",
        foam_header("volScalarField", "p", "0")
        + "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
        "boundaryField\n{\n"
        "    screw  { type zeroGradient; }\n    barrel { type zeroGradient; }\n"
        "    left   { type cyclic; }\n    right  { type cyclic; }\n"
        "    front  { type cyclic; }\n    back   { type cyclic; }\n}\n",
    )


def write_poiseuille_case(
    case: str,
    *,
    H: float,
    ny: int,
    K: float,
    n: float,
    G: float,
    gamma_min: float = 1.0e-2,
    end_time: int = 2000,
) -> None:
    """平行平板間のべき乗則 Poiseuille 流れ（x 方向に体積力 G、y に ny セル）.

    厳密解（h = H/2）:
        u(y) = n/(n+1) · (G/K)^(1/n) · [ h^((n+1)/n) − |y−h|^((n+1)/n) ]
    x は 1 セル cyclic、z は empty。
    """
    Lx = H  # 1 セルの幅。cyclic なので任意
    verts = [
        (0.0, 0.0, 0.0),
        (Lx, 0.0, 0.0),
        (Lx, H, 0.0),
        (0.0, H, 0.0),
        (0.0, 0.0, Lx),
        (Lx, 0.0, Lx),
        (Lx, H, Lx),
        (0.0, H, Lx),
    ]
    vtxt = "\n".join(f"    ({_f(x)} {_f(y)} {_f(z)})" for x, y, z in verts)
    _write(
        case,
        "system/blockMeshDict",
        foam_header("dictionary", "blockMeshDict", "system")
        + f"scale 1;\n\nvertices\n(\n{vtxt}\n);\n\n"
        f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) (1 {ny} 1) simpleGrading (1 1 1)\n);\n\n"
        "edges ();\n\nboundary\n(\n"
        "    bottom { type wall;   faces ((0 1 5 4)); }\n"
        "    top    { type wall;   faces ((3 7 6 2)); }\n"
        "    left   { type cyclic; neighbourPatch right; faces ((0 4 7 3)); }\n"
        "    right  { type cyclic; neighbourPatch left;  faces ((1 2 6 5)); }\n"
        "    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }\n"
        ");\n\nmergePatchPairs ();\n",
    )
    _write(case, "system/controlDict", _control_dict(end_time, end_time))
    _write(case, "system/fvSchemes", _fv_schemes())
    # Uy ≡ 0 なので正規化残差がノイズになり residualControl では止まらない。回数固定で回す
    _write(case, "system/fvSolution", _fv_solution(1e-9, 1e-10, residual_control=False))
    _write(case, "system/fvOptions", _fv_options((G, 0.0, 0.0)))
    _write(case, "constant/turbulenceProperties", _turbulence())
    _write(
        case,
        "constant/transportProperties",
        transport_properties(PowerLawViscosity(K=K, n=n, gamma_min=gamma_min)),
    )
    _write(
        case,
        "0/U",
        foam_header("volVectorField", "U", "0")
        + "dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   uniform (0 0 0);\n\n"
        "boundaryField\n{\n    bottom { type noSlip; }\n    top    { type noSlip; }\n"
        "    left   { type cyclic; }\n    right  { type cyclic; }\n"
        "    frontAndBack { type empty; }\n}\n",
    )
    _write(
        case,
        "0/p",
        foam_header("volScalarField", "p", "0")
        + "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
        "boundaryField\n{\n    bottom { type zeroGradient; }\n    top    { type zeroGradient; }\n"
        "    left   { type cyclic; }\n    right  { type cyclic; }\n"
        "    frontAndBack { type empty; }\n}\n",
    )


def powerlaw_poiseuille_exact(
    y: np.ndarray, *, H: float, K: float, n: float, G: float, gamma_min: float | None = None
) -> np.ndarray:
    """べき乗則 Poiseuille の厳密解（γ̇_min クランプ込み）.

    τ(y) = G·(h − y) に対し γ̇ = (τ/K)^(1/n)。クランプ μ_max = K·γ̇_min^(n−1) の
    領域（τ < τ_c = μ_max·γ̇_min）では γ̇ = τ/μ_max（ニュートン）に切り替わる。
    OpenFOAM の nuMax と ykep の gamma_min は同じ切り替えなので、厳密解にも
    同じクランプを入れておかないと「モデル差」と「解法差」が混ざる。
    """
    h = 0.5 * H
    m = (n + 1.0) / n
    s = np.abs(np.asarray(y, dtype=np.float64) - h)  # 中心からの距離
    a = n / (n + 1.0) * (G / K) ** (1.0 / n)
    u_pl = a * (h**m - s**m)
    if gamma_min is None:
        return u_pl
    mu_max = K * gamma_min ** (n - 1.0)
    s_c = mu_max * gamma_min / G  # クランプ領域の半幅
    if s_c >= h:
        return G / (2.0 * mu_max) * (h**2 - s**2)
    u_c = a * (h**m - s_c**m)
    u_newt = u_c + G / (2.0 * mu_max) * (s_c**2 - s**2)
    return np.where(s < s_c, u_newt, u_pl)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=["newtonian", "powerlaw"], required=True)
    ap.add_argument("--mu", type=float, default=1000.0)
    ap.add_argument("--K", type=float, default=2.0e4)
    ap.add_argument("--n", type=float, default=0.4)
    ap.add_argument("--G", type=float, default=DEFAULT_G)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = (
        NewtonianViscosity(mu=args.mu)
        if args.model == "newtonian"
        else PowerLawViscosity(K=args.K, n=args.n)
    )
    write_channel_case(args.out, DEFAULT_SPEC, args.G, model)
    xs, ys = x_sections(DEFAULT_SPEC), y_sections(DEFAULT_SPEC)
    print(
        f"wrote {args.out}: nx={sum(s.n for s in xs)} ny={sum(s.n for s in ys)} model={args.model} G={args.G:g}"
    )
    print(f"  force = ({-DEFAULT_SPEC.beta(args.G):.6g}, 0, {-args.G:.6g}) m/s^2 (rho=1)")
    print(f"  barrel U = ({DEFAULT_SPEC.u_barrel:.6g}, 0, {DEFAULT_SPEC.w_barrel:.6g}) m/s")
    print(f"  math.tan(phi) = {math.tan(DEFAULT_SPEC.phi):.6g}")


if __name__ == "__main__":
    main()

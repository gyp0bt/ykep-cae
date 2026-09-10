"""messi trama 蛇行流路の OpenFOAM ケース生成（nsb の独立検算）.

nsb が解いているのは「単位深さの 2 次元非圧縮 NS ＋ Brinkman 抗力」である。
連続式は div(ρ dy u, ρ dx v) = q_in − q_out で厚さ h の重みを持たず、h は
運動量式の抗力 12μ/h²·u にしか入らない（`nsb/assembly.py` の `drag`）。
したがって OpenFOAM 側は

    simpleFoam（非圧縮・層流） ＋ explicitPorositySource(DarcyForchheimer, d = 12/h²)

で同じ方程式になる。`explicitPorositySource` は名前に反して `eqn -= porosityEqn`
と行列ごと引くので抗力は対角に陰的に入り、rAU（＝ nsb の V/a_P）にも反映される。
ρ = 1000 の運動学的単位（simpleFoam は p/ρ を解く）で ν = μ/ρ = 3e-6 m²/s。

2 つの変種を作る:

- ``porous``: 領域 600×350 mm を丸ごと格子にし、流路セルに d = 12/h_channel²、
  閉塞セルに d = 12/h_blocked² を与える。**nsb の離散化そのもの**。
- ``walls``:  流路セルだけを切り出し（`subsetMesh`）、側壁を no-slip の実壁にして
  流路全体に d = 12/h_channel² を与える。**物理的に正しい深さ平均モデル**で、
  nsb の「閉塞を高抗力の栓で表す」近似がどれだけ効いているかの対照。

ポート（パターン左端 2 ノード）は半径 w/2 の円板セルを `subsetMesh` で刳り抜き、
露出面を inlet / outlet パッチにする。nsb はここをセル内の質量ソース／圧力シンクに
しているので、境界の与え方だけが違う（総流量は一致させる）。

使い方::

    python experiments/nsb/trama_of_case.py --variant porous --out /tmp/of-trama/porous
    # → 生成された run.sh を `of` ラッパで実行する（run_trama_of.sh 参照）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from trama_case import TramaGeometry, load_trama  # noqa: E402

DEFAULT_PATTERN = HERE.parents[2] / "tmp" / "pattern.json"


def _f(x: float) -> str:
    return f"{x:.12g}"


def _v(p: tuple[float, float, float]) -> str:
    return f"({_f(p[0])} {_f(p[1])} {_f(p[2])})"


def foam_header(cls: str, obj: str, location: str | None = None) -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n"
    )


def _write(case: str | Path, rel: str, body: str) -> None:
    path = Path(case) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@dataclass(frozen=True)
class OFCase:
    """生成したケースの諸元（yaml/json に落として突き合わせに使う）."""

    variant: str
    nx: int
    ny: int
    dx: float
    tz: float
    nu: float
    rho: float
    d_channel: float
    d_blocked: float
    flow_rate: float
    port_radius: float
    inlet: tuple[float, float]
    outlet: tuple[float, float]
    geo_variant: str = "orig"  # 中心線の作り方。突き合わせ側はこれを見て同じ geo を組む


# ----------------------------------------------------------------------
# topoSet: 流路 = 折れ線からの距離 <= w/2 = （区間シリンダ ∪ 頂点球）
# ----------------------------------------------------------------------
def channel_actions(geo: TramaGeometry, zc: float, name: str = "channel") -> str:
    """`distance_to_polyline <= w/2` をカプセル（シリンダ＋球）の和で表す topoSet アクション.

    セル中心は z = zc の 1 枚だけなので、軸を z = zc に置けばシリンダ／球の
    3 次元距離が面内距離と一致する（nsb の `make_trama_h` と同じ判定になる）。
    """
    r = geo.width / 2
    out = []
    first = True
    for a, b in zip(geo.polyline[:-1], geo.polyline[1:], strict=True):
        act = "new" if first else "add"
        first = False
        out.append(
            f"    {{ name {name}; type cellSet; action {act}; source cylinderToCell;\n"
            f"      point1 {_v((a[0], a[1], zc))}; point2 {_v((b[0], b[1], zc))};"
            f" radius {_f(r)}; }}"
        )
    for q in geo.polyline:
        out.append(
            f"    {{ name {name}; type cellSet; action add; source sphereToCell;\n"
            f"      origin {_v((q[0], q[1], zc))}; radius {_f(r)}; }}"
        )
    return "\n".join(out)


def port_actions(centre: tuple[float, float], radius: float, keep: str) -> str:
    """円板セルを `port` に集め、その補集合を `keep` にする（`subsetMesh keep` 用）."""
    return (
        "    { name port; type cellSet; action new; source cylinderToCell;\n"
        f"      point1 {_v((centre[0], centre[1], -1.0))};"
        f" point2 {_v((centre[0], centre[1], 1.0))}; radius {_f(radius)}; }}\n"
        f"    {{ name {keep}; type cellSet; action new; source cellToCell; set port; }}\n"
        f"    {{ name {keep}; type cellSet; action invert; }}"
    )


def topo_dict(actions: str) -> str:
    return foam_header("dictionary", "topoSetDict", "system") + f"actions\n(\n{actions}\n);\n"


# ----------------------------------------------------------------------
# system/*
# ----------------------------------------------------------------------
def control_dict(end_time: int, write_interval: int) -> str:
    return foam_header("dictionary", "controlDict", "system") + (
        "application     simpleFoam;\n"
        "startFrom       latestTime;\nstartTime       0;\nstopAt          endTime;\n"
        f"endTime         {end_time};\ndeltaT          1;\n"
        f"writeControl    timeStep;\nwriteInterval   {write_interval};\n"
        "purgeWrite      2;\nwriteFormat     ascii;\nwritePrecision  10;\n"
        "writeCompression off;\ntimeFormat      general;\nrunTimeModifiable false;\n\n"
        "functions\n{\n"
        "    massOut\n    {\n"
        "        type            surfaceFieldValue;\n"
        "        libs            (fieldFunctionObjects);\n"
        "        writeControl    timeStep;\n        writeInterval   50;\n"
        "        log             true;\n        writeFields     false;\n"
        "        regionType      patch;\n        name            outlet;\n"
        "        operation       sum;\n        fields          (phi);\n    }\n"
        "    massIn\n    {\n"
        "        type            surfaceFieldValue;\n"
        "        libs            (fieldFunctionObjects);\n"
        "        writeControl    timeStep;\n        writeInterval   50;\n"
        "        log             true;\n        writeFields     false;\n"
        "        regionType      patch;\n        name            inlet;\n"
        "        operation       sum;\n        fields          (phi);\n    }\n"
        "}\n"
    )


def control_dict_transient(
    end_time: float, write_interval: float, max_co: float, avg_start: float
) -> str:
    """pimpleFoam 用。Co 数で Δt を調整し、avg_start 以降の時間平均場も書く."""
    return foam_header("dictionary", "controlDict", "system") + (
        "application     pimpleFoam;\n"
        "startFrom       latestTime;\nstartTime       0;\nstopAt          endTime;\n"
        f"endTime         {_f(end_time)};\ndeltaT          1e-5;\n"
        f"writeControl    adjustableRunTime;\nwriteInterval   {_f(write_interval)};\n"
        "purgeWrite      3;\nwriteFormat     ascii;\nwritePrecision  10;\n"
        "writeCompression off;\ntimeFormat      general;\nrunTimeModifiable false;\n"
        f"adjustTimeStep  yes;\nmaxCo           {_f(max_co)};\nmaxDeltaT       1e-3;\n\n"
        "functions\n{\n"
        "    fieldAverage\n    {\n"
        "        type            fieldAverage;\n"
        "        libs            (fieldFunctionObjects);\n"
        "        writeControl    writeTime;\n"
        f"        timeStart       {_f(avg_start)};\n"
        "        fields\n        (\n"
        "            U { mean on; prime2Mean on; base time; }\n"
        "            p { mean on; prime2Mean on; base time; }\n"
        "        );\n    }\n"
        # 出口は p 固定なので、必要な圧力ヘッドは inlet 側で測る
        "    pIn\n    {\n"
        "        type            surfaceFieldValue;\n"
        "        libs            (fieldFunctionObjects);\n"
        "        writeControl    timeStep;\n        writeInterval   10;\n"
        "        log             true;\n        writeFields     false;\n"
        "        regionType      patch;\n        name            inlet;\n"
        "        operation       areaAverage;\n        fields          (p);\n    }\n"
        # volFieldValue は登録済みの場しか見ない。v2312 に maxMag 演算は無いので、
        # mag 関数オブジェクトで magU を作ってから max を取る
        "    magU\n    {\n"
        "        type            mag;\n"
        "        libs            (fieldFunctionObjects);\n"
        "        field           U;\n        result          magU;\n"
        "        executeControl  timeStep;\n        executeInterval 10;\n"
        "        writeControl    none;\n        log             false;\n    }\n"
        "    uMax\n    {\n"
        "        type            volFieldValue;\n"
        "        libs            (fieldFunctionObjects);\n"
        "        writeControl    timeStep;\n        writeInterval   10;\n"
        "        log             true;\n        writeFields     false;\n"
        "        regionType      all;\n"
        "        operation       max;\n        fields          (magU);\n    }\n"
        "}\n"
    )


def fv_solution_transient(*, n_outer: int, n_corr: int) -> str:
    return foam_header("dictionary", "fvSolution", "system") + (
        "solvers\n{\n"
        "    p\n    {\n        solver          GAMG;\n        smoother        DICGaussSeidel;\n"
        "        tolerance       1e-8;\n        relTol          0.01;\n"
        "        nCellsInCoarsestLevel 64;\n    }\n"
        "    pFinal\n    {\n        $p;\n        relTol          0;\n        tolerance       1e-9;\n    }\n"
        '    "(U|k|omega)"\n    {\n        solver          PBiCGStab;\n'
        "        preconditioner  DILU;\n        tolerance       1e-9;\n        relTol          0.01;\n    }\n"
        '    "(U|k|omega)Final"\n    {\n        $U;\n        relTol          0;\n    }\n}\n\n'
        "PIMPLE\n{\n"
        f"    nOuterCorrectors {n_outer};\n    nCorrectors      {n_corr};\n"
        "    nNonOrthogonalCorrectors 0;\n"
        "    residualControl\n    {\n"
        '        "(U|p)"  { tolerance 1e-4; relTol 0; }\n    }\n}\n\n'
        "relaxationFactors\n{\n"
        '    equations { ".*" 1.0; ".*Final" 1.0; }\n'
        '    fields    { ".*" 1.0; }\n}\n'
    )


def fv_schemes(scheme: str, limited_grad: bool, transient: bool = False) -> str:
    b = "" if transient else "bounded "
    div = {
        "sou": f"{b}Gauss linearUpwind limitedGrad",
        "upwind": f"{b}Gauss upwind",
        "linear": f"{b}Gauss linear",
    }[scheme]
    ddt = "Euler" if transient else "steadyState"
    grad = (
        "gradSchemes\n{\n    default         Gauss linear;\n"
        "    limitedGrad     cellLimited Gauss linear 1;\n}\n"
        if limited_grad
        else "gradSchemes\n{\n    default         Gauss linear;\n"
        "    limitedGrad     Gauss linear;\n}\n"
    )
    return (
        foam_header("dictionary", "fvSchemes", "system")
        + f"ddtSchemes      {{ default {ddt}; }}\n"
        + grad
        + "divSchemes\n{\n    default         none;\n"
        f"    div(phi,U)      {div};\n"
        "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\n"
        "laplacianSchemes { default Gauss linear corrected; }\n"
        "interpolationSchemes { default linear; }\n"
        "snGradSchemes   { default corrected; }\n"
        "wallDist        { method meshWave; }\n"
    )


def fv_solution(
    *, relax_u: float, relax_p: float, consistent: bool, p_tol: float, u_tol: float
) -> str:
    return foam_header("dictionary", "fvSolution", "system") + (
        "solvers\n{\n"
        "    p\n    {\n        solver          GAMG;\n        smoother        DICGaussSeidel;\n"
        "        tolerance       1e-9;\n        relTol          0.01;\n"
        "        nPreSweeps      0;\n        nPostSweeps     2;\n"
        "        cacheAgglomeration true;\n        nCellsInCoarsestLevel 64;\n"
        "        agglomerator    faceAreaPair;\n        mergeLevels     1;\n    }\n"
        "    U\n    {\n        solver          PBiCGStab;\n        preconditioner  DILU;\n"
        "        tolerance       1e-10;\n        relTol          0.01;\n    }\n}\n\n"
        "SIMPLE\n{\n    nNonOrthogonalCorrectors 0;\n"
        f"    consistent      {'yes' if consistent else 'no'};\n"
        f"    residualControl {{ p {_f(p_tol)}; U {_f(u_tol)}; }}\n}}\n\n"
        "relaxationFactors\n{\n"
        f"    equations {{ U {_f(relax_u)}; }}\n"
        f"    fields    {{ p {_f(relax_p)}; }}\n}}\n"
    )


def fv_options(case_spec: OFCase) -> str:
    """Brinkman 抗力 12μ/h² u → DarcyForchheimer d = 12/h²（ν 倍されて m/s² になる）."""

    def src(name: str, d: float, zone: str) -> str:
        sel = f"        selectionMode   cellZone;\n        cellZone        {zone};\n"
        return (
            f"{name}\n{{\n"
            "    type            explicitPorositySource;\n"
            "    active          yes;\n\n"
            "    explicitPorositySourceCoeffs\n    {\n"
            + sel
            + "        type            DarcyForchheimer;\n"
            f"        d               ({_f(d)} {_f(d)} {_f(d)});\n"
            "        f               (0 0 0);\n"
            "        coordinateSystem\n        {\n"
            "            type            cartesian;\n"
            "            origin          (0 0 0);\n"
            "            rotation        { type axes; e1 (1 0 0); e2 (0 1 0); }\n"
            "        }\n    }\n}\n"
        )

    body = foam_header("dictionary", "fvOptions", "system")
    if case_spec.variant == "porous":
        body += src("gapDragChannel", case_spec.d_channel, "channel")
        body += "\n" + src("gapDragBlocked", case_spec.d_blocked, "blocked")
    else:
        body += src("gapDrag", case_spec.d_channel, "channel")
    return body


# ----------------------------------------------------------------------
# 0/*
# ----------------------------------------------------------------------
def field_u(patches: dict[str, str], flow_rate: float) -> str:
    lines = []
    for name, kind in patches.items():
        if name == "inlet":
            lines.append(
                f"    {name}\n    {{\n        type            flowRateInletVelocity;\n"
                f"        volumetricFlowRate constant {_f(flow_rate)};\n"
                "        value           uniform (0 0 0);\n    }"
            )
        elif name == "outlet":
            lines.append(
                f"    {name}\n    {{\n        type            inletOutlet;\n"
                "        inletValue      uniform (0 0 0);\n"
                "        value           uniform (0 0 0);\n    }"
            )
        elif kind == "empty":
            lines.append(f"    {name} {{ type empty; }}")
        else:
            lines.append(f"    {name} {{ type noSlip; }}")
    return (
        foam_header("volVectorField", "U", "0")
        + "dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   uniform (0 0 0);\n\n"
        + "boundaryField\n{\n"
        + "\n".join(lines)
        + "\n}\n"
    )


def field_p(patches: dict[str, str]) -> str:
    lines = []
    for name, kind in patches.items():
        if name == "outlet":
            lines.append(
                f"    {name}\n    {{\n        type            fixedValue;\n"
                "        value           uniform 0;\n    }"
            )
        elif kind == "empty":
            lines.append(f"    {name} {{ type empty; }}")
        else:
            lines.append(f"    {name} {{ type zeroGradient; }}")
    return (
        foam_header("volScalarField", "p", "0")
        + "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
        + "boundaryField\n{\n"
        + "\n".join(lines)
        + "\n}\n"
    )


# ----------------------------------------------------------------------
def write_case(
    case: str | Path,
    geo: TramaGeometry,
    *,
    variant: str = "porous",
    dx_mm: float = 1.5,
    mass_flow: float = 0.15,
    rho: float = 1000.0,
    mu: float = 3.0e-3,
    h_channel: float = 3.8e-3,
    h_blocked: float = 1.0e-5,
    end_time: int = 5000,
    write_interval: int = 500,
    scheme: str = "sou",
    limited_grad: bool = True,
    relax_u: float = 0.7,
    relax_p: float = 0.3,
    consistent: bool = False,
    p_tol: float = 1.0e-6,
    u_tol: float = 1.0e-7,
    transient: bool = False,
    end_time_s: float = 4.0,
    write_interval_s: float = 0.05,
    max_co: float = 5.0,
    avg_start: float = 2.0,
    n_outer: int = 3,
    n_corr: int = 2,
    port_shrink_cells: float = 2.0,
) -> OFCase:
    """ケース一式 + メッシュ生成手順 `mesh.sh` を書き出す."""
    if variant not in ("porous", "walls"):
        raise ValueError(f"variant は porous / walls: {variant!r}")
    case = Path(case)
    nx = int(round(geo.lx / (dx_mm * 1e-3)))
    ny = int(round(geo.ly / (dx_mm * 1e-3)))
    tz = h_channel  # 2D 格子の厚み。この値なら体積流量が mass_flow/rho にそのまま一致する
    zc = 0.5 * tz
    # ポートの刳り抜き半径。w/2 ちょうどだと円周が流路と閉塞域の境目に乗り、
    # porous 変種では入口の外周が栓に接して そこへ流量を押し込む（圧力が 3.6 MPa まで跳ねる）。
    # 2 セル分縮めて円周を流路の内側に置く。
    r_port = geo.width / 2 - port_shrink_cells * (geo.lx / nx)
    if r_port <= 2 * (geo.lx / nx):
        raise ValueError(f"ポート半径が小さすぎる: {r_port:g} m")
    spec = OFCase(
        variant=variant,
        nx=nx,
        ny=ny,
        dx=geo.lx / nx,
        tz=tz,
        nu=mu / rho,
        rho=rho,
        d_channel=12.0 / h_channel**2,
        d_blocked=12.0 / h_blocked**2,
        flow_rate=mass_flow / rho,
        port_radius=r_port,
        inlet=(float(geo.inlet[0]), float(geo.inlet[1])),
        outlet=(float(geo.outlet[0]), float(geo.outlet[1])),
        geo_variant=geo.variant,
    )

    # --- blockMesh -----------------------------------------------------
    verts = [
        (0.0, 0.0, 0.0),
        (geo.lx, 0.0, 0.0),
        (geo.lx, geo.ly, 0.0),
        (0.0, geo.ly, 0.0),
        (0.0, 0.0, tz),
        (geo.lx, 0.0, tz),
        (geo.lx, geo.ly, tz),
        (0.0, geo.ly, tz),
    ]
    vtxt = "\n".join(f"    {_v(p)}" for p in verts)
    _write(
        case,
        "system/blockMeshDict",
        foam_header("dictionary", "blockMeshDict", "system")
        + f"scale 1;\n\nvertices\n(\n{vtxt}\n);\n\n"
        f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)\n);\n\n"
        "edges ();\n\nboundary\n(\n"
        "    box   { type wall; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2)); }\n"
        "    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }\n"
        ");\n\nmergePatchPairs ();\n",
    )

    # --- topoSet 段（メッシュを削る順に 1 枚ずつ） -----------------------
    r = spec.port_radius
    stages: list[tuple[str, str, str]] = []  # (dict 名, subsetMesh の set, -patch 名)
    if variant == "walls":
        _write(case, "system/topoSetDict.channel", topo_dict(channel_actions(geo, zc)))
        stages.append(("topoSetDict.channel", "channel", "wallChannel"))
    _write(case, "system/topoSetDict.portIn", topo_dict(port_actions(spec.inlet, r, "keepIn")))
    stages.append(("topoSetDict.portIn", "keepIn", "inlet"))
    _write(case, "system/topoSetDict.portOut", topo_dict(port_actions(spec.outlet, r, "keepOut")))
    stages.append(("topoSetDict.portOut", "keepOut", "outlet"))
    if variant == "walls":
        acts = (
            "    { name channel; type cellSet; action new; source boxToCell;\n"
            "      box (-1 -1 -1) (1 1 1); }\n"
            "    { name channel; type cellZoneSet; action new; source setToCellZone;"
            " set channel; }"
        )
        _write(case, "system/topoSetDict.zones", topo_dict(acts))
    if variant == "porous":
        acts = (
            channel_actions(geo, zc)
            + "\n    { name channel; type cellZoneSet; action new; source setToCellZone;"
            " set channel; }\n"
            "    { name blocked; type cellSet; action new; source cellToCell; set channel; }\n"
            "    { name blocked; type cellSet; action invert; }\n"
            "    { name blocked; type cellZoneSet; action new; source setToCellZone;"
            " set blocked; }"
        )
        _write(case, "system/topoSetDict.zones", topo_dict(acts))

    # subsetMesh は時刻ディレクトリのフィールドも部分集合に写して書き戻すので、
    # 初期場は 0.orig に置き、メッシュが出来てから 0 にコピーする（OpenFOAM の定石）。
    mesh_sh = ["#!/bin/sh", "set -e", "rm -rf constant/polyMesh 0", "blockMesh"]
    for dict_name, keep, patch in stages:
        mesh_sh += [
            "rm -rf constant/polyMesh/sets",
            f"topoSet -dict system/{dict_name}",
            f"subsetMesh {keep} -patch {patch} -overwrite",
        ]
    mesh_sh += ["rm -rf constant/polyMesh/sets", "topoSet -dict system/topoSetDict.zones"]
    mesh_sh += ["checkMesh -constant | tail -40"]
    _write(case, "mesh.sh", "\n".join(mesh_sh) + "\n")
    os.chmod(case / "mesh.sh", 0o755)

    # --- 物性・スキーム ------------------------------------------------
    _write(
        case,
        "constant/transportProperties",
        foam_header("dictionary", "transportProperties", "constant")
        + f"transportModel  Newtonian;\n\nnu              {_f(spec.nu)};\n",
    )
    _write(
        case,
        "constant/turbulenceProperties",
        foam_header("dictionary", "turbulenceProperties", "constant")
        + "simulationType  laminar;\n",
    )
    if transient:
        _write(
            case,
            "system/controlDict",
            control_dict_transient(end_time_s, write_interval_s, max_co, avg_start),
        )
        _write(case, "system/fvSolution", fv_solution_transient(n_outer=n_outer, n_corr=n_corr))
    else:
        _write(case, "system/controlDict", control_dict(end_time, write_interval))
        _write(
            case,
            "system/fvSolution",
            fv_solution(
                relax_u=relax_u, relax_p=relax_p, consistent=consistent, p_tol=p_tol, u_tol=u_tol
            ),
        )
    _write(case, "system/fvSchemes", fv_schemes(scheme, limited_grad, transient))
    _write(case, "system/fvOptions", fv_options(spec))

    patches: dict[str, str] = {}
    if variant == "porous":
        patches["box"] = "wall"
    else:
        patches["box"] = "wall"
        patches["wallChannel"] = "wall"
    patches["inlet"] = "patch"
    patches["outlet"] = "patch"
    patches["frontAndBack"] = "empty"
    _write(case, "0.orig/U", field_u(patches, spec.flow_rate))
    _write(case, "0.orig/p", field_p(patches))
    _write(case, "case.json", json.dumps(spec.__dict__, indent=1) + "\n")
    return spec


# ----------------------------------------------------------------------
def retype_patches(case: str | Path, kinds: dict[str, str]) -> None:
    """`constant/polyMesh/boundary` のパッチ型を書き換える.

    `subsetMesh -patch <name>` は露出面を **empty 型**の新パッチに入れる。empty のまま
    だと `polyMesh::calcDirections` が x, y 方向まで「空」と判定して 2 次元解が立たない
    （checkMesh が "0 geometric directions" と言う）ので、inlet/outlet を patch、
    刳り抜きで出来た側壁を wall に直す。`inGroups` も型に合わせて消す。
    """
    path = Path(case) / "constant" / "polyMesh" / "boundary"
    text = path.read_text(encoding="utf-8")
    for name, kind in kinds.items():
        pat = re.compile(rf"(\n\s*{re.escape(name)}\s*\n\s*\{{)(.*?)(\n\s*\}})", re.S)
        m = pat.search(text)
        if m is None:
            raise KeyError(f"パッチ {name} が boundary に無い: {path}")
        body = re.sub(r"\btype\s+\w+\s*;", f"type            {kind};", m.group(2))
        body = re.sub(r"\n\s*inGroups[^;]*;", "", body)
        if kind == "wall":
            body = body.replace(
                f"type            {kind};",
                f"type            {kind};\n        inGroups        1(wall);",
            )
        text = text[: m.start(2)] + body + text[m.end(2) :]
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pattern", nargs="?", type=Path, default=DEFAULT_PATTERN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["porous", "walls"], default="porous")
    ap.add_argument(
        "--geo-variant",
        choices=["orig", "ortho", "lead"],
        default="orig",
        help="中心線の作り方（nsb 側の trama_case.py --variant と揃える）",
    )
    ap.add_argument("--dx", type=float, default=1.5, help="格子幅 [mm]")
    ap.add_argument("--mass", type=float, default=0.15, help="質量流量 [kg/s]")
    ap.add_argument("--mu", type=float, default=3.0e-3)
    ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--h-channel", type=float, default=3.8e-3)
    ap.add_argument("--h-blocked", type=float, default=1.0e-5)
    ap.add_argument("--end-time", type=int, default=5000)
    ap.add_argument("--write-interval", type=int, default=500)
    ap.add_argument("--scheme", choices=["sou", "upwind", "linear"], default="sou")
    ap.add_argument("--no-limited-grad", action="store_true")
    ap.add_argument("--relax-u", type=float, default=0.7)
    ap.add_argument("--relax-p", type=float, default=0.3)
    ap.add_argument("--simplec", action="store_true")
    a = ap.parse_args()

    geo = load_trama(a.pattern, variant=a.geo_variant)
    spec = write_case(
        a.out,
        geo,
        variant=a.variant,
        dx_mm=a.dx,
        mass_flow=a.mass,
        rho=a.rho,
        mu=a.mu,
        h_channel=a.h_channel,
        h_blocked=a.h_blocked,
        end_time=a.end_time,
        write_interval=a.write_interval,
        scheme=a.scheme,
        limited_grad=not a.no_limited_grad,
        relax_u=a.relax_u,
        relax_p=a.relax_p,
        consistent=a.simplec,
    )
    print(f"wrote {a.out}: variant={spec.variant} {spec.nx}x{spec.ny} dx={spec.dx * 1e3:.3g} mm")
    print(f"  nu={spec.nu:g} m^2/s  Q={spec.flow_rate:g} m^3/s (tz={spec.tz:g} m)")
    print(f"  d_channel={spec.d_channel:.6g} 1/m^2  d_blocked={spec.d_blocked:.6g} 1/m^2")
    print(
        f"  nu*d: channel {spec.nu * spec.d_channel:.6g} 1/s, blocked {spec.nu * spec.d_blocked:.6g} 1/s"
    )
    print(f"  ports r={spec.port_radius * 1e3:.4g} mm at {spec.inlet} / {spec.outlet}")


if __name__ == "__main__":
    main()

"""押出断面のフィールド図ページを組み立てる.

実解をラスタ化して base64 PNG に焼き、SVG のオーバーレイ（フライト輪郭・流線・
寸法）を重ねた 1 枚ページを吐く。
"""

from __future__ import annotations

import base64
import io
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from xkep_cae_fluid.extruder import (
    ExtruderFlowProcess,
    NewtonianViscosity,
    PowerLawViscosity,
    ScrewSpec,
)
from xkep_cae_fluid.extruder.data import ExtruderFlowInput

OUT = os.path.dirname(os.path.abspath(__file__))
G_BACK = 5.0e6

SPEC = ScrewSpec(
    D=0.040, lead=0.040, H=0.004, e=0.004, delta=1.0e-4, N=100 / 60,
    nx_channel=180, nx_land=44, ny_bulk=56, n_gap=18,
)


def solve(model):
    p = ExtruderFlowProcess()
    p.viscosity = model
    return p.process(ExtruderFlowInput(spec=SPEC, G=G_BACK))


fN = solve(NewtonianViscosity(mu=1000.0))
fP = solve(PowerLawViscosity(K=2e4, n=0.4))
g = fN.grid
s = g.spec
fluid = ~g.solid
COS, SIN = math.cos(s.phi), math.sin(s.phi)
AXIAL = fN.u * COS + fN.w * SIN


def make_raster(field, x0, x1, y0, y1, nx, ny):
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    it = RegularGridInterpolator(
        (g.xc, g.yc), field, bounds_error=False, fill_value=None, method="linear"
    )
    return it(np.stack([xx.ravel(), yy.ravel()], -1)).reshape(nx, ny)


def png(field, vmin, vmax, box, size, *, mode="lin", cmap="viridis"):
    x0, x1, y0, y1 = box
    nx, ny = size
    a = make_raster(field, x0, x1, y0, y1, nx, ny)
    solid = make_raster(g.solid.astype(float), x0, x1, y0, y1, nx, ny) > 0.5
    if mode == "log":
        a = np.clip(a, vmin, vmax)
        norm = LogNorm(vmin, vmax)
    elif mode == "div":
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin, vmax)
    rgba = (colormaps[cmap](norm(a)) * 255).astype(np.uint8)
    rgba[solid] = [0, 0, 0, 0]
    img = Image.fromarray(np.transpose(rgba, (1, 0, 2))[::-1], "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def bar(cmap, mode="lin"):
    """横長カラーバーを PNG で作る."""
    t = np.linspace(0, 1, 512)
    if mode == "log":
        t = np.linspace(0, 1, 512)
    rgba = (colormaps[cmap](t) * 255).astype(np.uint8)
    img = Image.fromarray(rgba[None, :, :].repeat(16, 0), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


FULL = (0.0, s.W_t, 0.0, s.H)
X_MID = s.W_t / 2
ZOOM = (X_MID - 0.005, X_MID + 0.005, s.H - 0.0005, s.H)

w_max = float(fN.w[fluid].max())
ax_max = float(AXIAL[fluid].max())
gam_max = float(fP.gamma_dot[fluid].max())
mu_min = float(fP.mu[fluid].min())
mu_max = float(fP.mu[fluid].max())

PANELS = [
    dict(
        key="u", title="横断方向の速度 u", sub="Newtonian μ=1000 Pa·s",
        img=png(fN.u, -0.075, 0.075, FULL, (1400, 300), mode="div", cmap="RdBu_r"),
        cbar=bar("RdBu_r"), lo="−0.075", mid="0", hi="+0.075", unit="m/s",
        note="バレルに引かれて上層は左（青）、底を右（赤）へ戻る。この往復が断面内の渦になり、"
             "材料を何度もバレル直下の高せん断域へ連れ戻す。混練の主機構はこれ。",
        streams=True,
    ),
    dict(
        key="w", title="下流方向の速度 w", sub="Newtonian μ=1000 Pa·s",
        img=png(fN.w, 0.0, w_max, FULL, (1400, 300), cmap="magma"),
        cbar=bar("magma"), lo="0", mid="", hi=f"{w_max:.3f}", unit="m/s",
        note="バレル直下が最速（+V cosφ = 0.200 m/s）。隙間の中も明るいことに注目——"
             "隙間はいちばん速い層に追加の通り道を空けている。だから断面流量 Q はむしろ増える。",
        streams=False,
    ),
    dict(
        key="axial", title="軸方向の速度 u cosφ + w sinφ", sub="＝実際に機械の出口へ向かう成分",
        img=png(AXIAL, -0.004, ax_max, FULL, (1400, 300), mode="div", cmap="PuOr_r"),
        cbar=bar("PuOr_r"), lo="−0.004", mid="0", hi=f"{ax_max:.3f}", unit="m/s",
        note="ここが押出の本質。バレル面（最上部）でちょうどゼロ——バレルは周方向にしか"
             "動かないので、貼りついた材料は前へ進まない。隙間の中は薄紫（負）で、"
             "漏れは軸方向に後戻りしている。滞留時間の長い裾はこの 2 か所が作る。",
        streams=False,
    ),
    dict(
        key="gamma", title="せん断速度 γ̇", sub="Power law K=2×10⁴, n=0.4（対数目盛）",
        img=png(fP.gamma_dot, 5.0, 3000.0, FULL, (1400, 300), mode="log", cmap="inferno"),
        cbar=bar("inferno"), lo="5", mid="120", hi="3000", unit="1/s",
        note=f"隙間で最大 {gam_max:.0f} 1/s、溝の中央は 99 1/s。実に 28 倍。"
             "0.1 mm の隙間をバレルが 0.2 m/s で擦っていくのだから当然だが、"
             "この一点が粘度も発熱も支配してしまう。",
        streams=False,
    ),
    dict(
        key="mu", title="粘度 μ", sub="Power law（対数目盛）— せん断減粘",
        img=png(fP.mu, 150.0, 30000.0, FULL, (1400, 300), mode="log", cmap="cividis"),
        cbar=bar("cividis"), lo="150", mid="2000", hi="30000", unit="Pa·s",
        note=f"同じ断面の中で {mu_min:.0f} 〜 {mu_max:.0f} Pa·s。**{mu_max/mu_min:.0f} 倍**の差。"
             "隙間の樹脂は溝の底より桁違いに柔らかい。だから漏れは粘度一定で"
             "見積もるより起きやすく、押出量は 39.5 → 34.6 kg/h に落ちる。",
        streams=False,
    ),
]

ZOOMS = [
    dict(
        key="zgamma", title="隙間の拡大：せん断速度",
        img=png(fP.gamma_dot, 5.0, 3000.0, ZOOM, (1200, 240), mode="log", cmap="inferno"),
        note="横 10 mm × 縦 0.5 mm（縦を 24 倍に誇張）。フライトの頂と"
             "バレルの間、わずか 0.1 mm の層だけが白熱している。",
    ),
    dict(
        key="zmu", title="隙間の拡大：粘度",
        img=png(fP.mu, 150.0, 30000.0, ZOOM, (1200, 240), mode="log", cmap="cividis"),
        note="同じ場所の粘度。隙間だけが濃紺（＝柔らかい）に落ちている。"
             "せん断減粘が漏れを増やす経路がここに見える。",
    ),
]

# --- 流線（ψ 等高線） ---
xn = np.concatenate([[0], np.cumsum(g.dx)])
yn = np.concatenate([[0], np.cumsum(g.dy)])
psi = fN.psi[: g.nx + 1, :]
fig = plt.figure()
cs = plt.contour(xn, yn, psi.T, levels=np.linspace(float(psi.min()), float(psi.max()), 34))
plt.close(fig)

VBW, VBH = 1400.0, 300.0
stream_paths = []
for segs in cs.allsegs:
    for seg in segs:
        if len(seg) < 8:
            continue
        step = max(1, len(seg) // 110)
        pts = seg[::step]
        d = " ".join(
            f"{px / s.W_t * VBW:.1f},{VBH - py / s.H * VBH:.1f}" for px, py in pts
        )
        stream_paths.append(d)

# ---------------------------------------------------------------- HTML
def overlay(show_streams: bool) -> str:
    """フライト輪郭・壁・（任意で）流線の SVG オーバーレイ."""
    xl = (s.W_t - s.e) / 2 / s.W_t * VBW
    xr = (s.W_t + s.e) / 2 / s.W_t * VBW
    yt = VBH - (s.H - s.delta) / s.H * VBH
    st = ""
    if show_streams:
        st = "\n".join(
            f'<polyline points="{d}" fill="none" stroke="#EAF2F0" '
            f'stroke-width="1.1" opacity=".38"/>'
            for d in stream_paths
        )
    return f"""<svg class="ov" viewBox="0 0 {VBW:.0f} {VBH:.0f}" preserveAspectRatio="none" aria-hidden="true">
{st}
<rect x="{xl:.1f}" y="{yt:.1f}" width="{xr - xl:.1f}" height="{VBH - yt:.1f}"
      fill="none" stroke="#F2C9A8" stroke-width="2"/>
<line x1="0" y1="1" x2="{VBW:.0f}" y2="1" stroke="#F2C9A8" stroke-width="2.5"/>
<line x1="0" y1="{VBH - 1:.0f}" x2="{VBW:.0f}" y2="{VBH - 1:.0f}" stroke="#8A9490" stroke-width="2.5"/>
</svg>"""


def panel_html(p) -> str:
    return f"""<section class="panel">
  <header class="ph">
    <h2>{p['title']}</h2>
    <p class="sub">{p['sub']}</p>
  </header>
  <div class="plot">
    <img src="data:image/png;base64,{p['img']}" alt="{p['title']}の分布">
    {overlay(p['streams'])}
    <span class="tag tag-tl">バレル（移動壁）</span>
    <span class="tag tag-bl">スクリュー根元</span>
    <span class="tag tag-fl">フライト</span>
  </div>
  <div class="cb">
    <span class="v">{p['lo']}</span>
    <img class="cbimg" src="data:image/png;base64,{p['cbar']}" alt="">
    <span class="v">{p['hi']} <em>{p['unit']}</em></span>
  </div>
  <p class="note">{p['note']}</p>
</section>"""


def zoom_html(z) -> str:
    return f"""<div class="zoom">
  <h3>{z['title']}</h3>
  <div class="plot zplot"><img src="data:image/png;base64,{z['img']}" alt="{z['title']}"></div>
  <p class="note">{z['note']}</p>
</div>"""


HTML = f"""<title>押出断面のフィールド</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Zen+Kaku+Gothic+New:wght@400;500;700&display=swap">
<style>
:root{{
  --bg:#0C1113; --panel:#131A1C; --edge:#1F292B; --line:#2A3638;
  --fg:#E8F0EE; --dim:#8D9C99; --hot:#F2955B; --cool:#5CC6D0;
  --f-d:"Archivo","Zen Kaku Gothic New",system-ui,sans-serif;
  --f-b:"Zen Kaku Gothic New","Archivo",system-ui,sans-serif;
  --f-m:"IBM Plex Mono",ui-monospace,monospace;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:var(--f-b);
  font-size:16px;line-height:1.8;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:78rem;margin:0 auto;padding:0 1.25rem 5rem}}

header.top{{padding:4rem 0 2rem;border-bottom:1px solid var(--line)}}
.kicker{{font-family:var(--f-m);font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;
  color:var(--hot);margin:0 0 .9rem}}
h1{{font-family:var(--f-d);font-weight:700;font-size:clamp(2.1rem,5.5vw,3.2rem);
  line-height:1.08;letter-spacing:-.02em;margin:0 0 .9rem;text-wrap:balance}}
.lede{{max-width:40rem;color:var(--dim);margin:0;font-size:1rem}}
.meta{{font-family:var(--f-m);font-size:.74rem;color:var(--dim);
  display:flex;flex-wrap:wrap;gap:.35rem 1.6rem;margin-top:1.4rem}}
.meta b{{color:var(--fg);font-weight:500}}

.panel{{margin:3.2rem 0 0;padding-top:2.2rem;border-top:1px solid var(--line)}}
.panel:first-of-type{{border-top:none}}
.ph{{display:flex;flex-wrap:wrap;align-items:baseline;gap:.4rem 1.1rem;margin-bottom:1rem}}
h2{{font-family:var(--f-d);font-weight:600;font-size:1.35rem;margin:0;letter-spacing:-.005em}}
.sub{{font-family:var(--f-m);font-size:.76rem;color:var(--dim);margin:0}}

.plot{{position:relative;line-height:0;border:1px solid var(--edge);border-radius:2px;
  background:
    repeating-linear-gradient(45deg,#222C2E 0 3px,#1A2325 3px 6px);
  overflow:hidden}}
.plot img{{display:block;width:100%;height:auto}}
.ov{{position:absolute;inset:0;width:100%;height:100%}}
.tag{{position:absolute;font-family:var(--f-m);font-size:.68rem;letter-spacing:.04em;
  color:#0C1113;background:#DDE7E4;padding:.1rem .4rem;border-radius:2px;line-height:1.5}}
.tag-tl{{top:.35rem;left:.45rem}}
.tag-bl{{bottom:.35rem;left:.45rem}}
.tag-fl{{bottom:.35rem;left:50%;transform:translateX(-50%);background:#F2C9A8}}

.cb{{display:flex;align-items:center;gap:.7rem;margin:.7rem 0 0;
  font-family:var(--f-m);font-size:.74rem;color:var(--dim)}}
.cbimg{{flex:1;height:9px;width:auto;display:block;border-radius:1px;min-width:6rem}}
.cb .v{{white-space:nowrap;font-variant-numeric:tabular-nums}}
.cb em{{font-style:normal;color:var(--dim);opacity:.75}}

.note{{max-width:44rem;margin:1rem 0 0;font-size:.94rem;color:#C4D2CF;line-height:1.85}}
.note strong{{color:var(--hot);font-weight:700}}

.zooms{{display:grid;gap:2rem;margin-top:2.2rem}}
@media(min-width:60rem){{.zooms{{grid-template-columns:1fr 1fr}}}}
.zoom h3{{font-family:var(--f-d);font-weight:600;font-size:1rem;margin:0 0 .7rem}}
.zplot{{background:repeating-linear-gradient(45deg,#222C2E 0 3px,#1A2325 3px 6px)}}
.zoom .note{{font-size:.88rem;margin-top:.7rem}}

.readme{{margin:3.4rem 0 0;padding-top:2rem;border-top:1px solid var(--line);
  display:grid;gap:1.6rem}}
@media(min-width:52rem){{.readme{{grid-template-columns:repeat(3,1fr)}}}}
.card{{background:var(--panel);border:1px solid var(--edge);border-radius:3px;padding:1.1rem 1.2rem}}
.card h4{{font-family:var(--f-m);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--hot);margin:0 0 .5rem;font-weight:500}}
.card p{{margin:0;font-size:.88rem;color:#C4D2CF;line-height:1.8}}
.card .big{{font-family:var(--f-m);font-size:1.5rem;color:var(--fg);
  font-variant-numeric:tabular-nums;display:block;line-height:1.4}}

footer{{margin-top:3.4rem;padding-top:1.4rem;border-top:1px solid var(--line);
  font-family:var(--f-m);font-size:.73rem;color:var(--dim);line-height:1.9}}
a{{color:var(--hot)}}
:focus-visible{{outline:2px solid var(--hot);outline-offset:2px}}
@media(max-width:34rem){{body{{font-size:15px}} .tag{{font-size:.6rem}}}}
</style>

<div class="wrap">
<header class="top">
  <p class="kicker">40 mm 単軸押出機 / 計量部 / 背圧 5 MPa/m</p>
  <h1>押出断面のフィールド</h1>
  <p class="lede">
    螺旋の溝を平面に展開した 1 枚の断面。横 38.1 mm × 深さ 4 mm、
    真ん中にフライト（幅 4 mm）が立ち、その頂とバレルの間に 0.1 mm の隙間がある。
    ここに何が起きているのかを、実際に解いた場でそのまま見る。
  </p>
  <p class="meta">
    <span>格子 <b>{g.nx}×{g.ny}</b>（{g.nx * g.ny:,} セル）</span>
    <span>ニュートン解 <b>{fN.elapsed_seconds:.2f} s</b></span>
    <span>べき乗則解 <b>{fP.elapsed_seconds:.1f} s</b>（{fP.n_iter} 反復）</span>
    <span>縦方向を <b>約 5 倍</b>に誇張して表示</span>
  </p>
</header>

{"".join(panel_html(p) for p in PANELS)}

<section class="panel">
  <header class="ph">
    <h2>隙間だけを拡大する</h2>
    <p class="sub">横 10 mm × 縦 0.5 mm — 縦を 24 倍に誇張</p>
  </header>
  <div class="zooms">{"".join(zoom_html(z) for z in ZOOMS)}</div>
</section>

<section class="readme">
  <div class="card">
    <h4>この 3 枚で押出が読める</h4>
    <p>横断方向 u が<strong>渦</strong>を作って混ぜ、下流方向 w が<strong>運び</strong>、
    その斜めの合成である軸方向成分だけが<strong>本当に出口へ向かう</strong>。
    3 枚目でバレル面と隙間が「進まない場所」として浮かび上がる。</p>
  </div>
  <div class="card">
    <h4>隙間のせん断速度</h4>
    <span class="big">{gam_max:.0f} <span style="font-size:.8rem;color:var(--dim)">1/s</span></span>
    <p>溝中央（99 1/s）の <strong>{gam_max / 99:.0f} 倍</strong>。
    0.1 mm の層をバレルが 0.2 m/s で擦る、ただそれだけの結果。</p>
  </div>
  <div class="card">
    <h4>断面内の粘度差</h4>
    <span class="big">{mu_max / mu_min:.0f} <span style="font-size:.8rem;color:var(--dim)">倍</span></span>
    <p>{mu_min:.0f} 〜 {mu_max:.0f} Pa·s。同じ樹脂・同じ断面でこれだけ違う。
    「樹脂の粘度は何 Pa·s か」という問いが成立しない理由。</p>
  </div>
</section>

<footer>
  <p>
    計算: xkep-cae-fluid / extruder。断面 2.5D（展開チャネル・クリープ流れ・疎直接解）。<br>
    条件: D=40 mm・リード 40 mm・H=4 mm・e=4 mm・δ=0.1 mm・N=100 rpm・dp/dz=5 MPa/m。
    ニュートン μ=1000 Pa·s / べき乗則 K=2×10⁴ Pa·sⁿ, n=0.4。<br>
    白い細線は流れ関数 ψ の等高線（＝断面内の流線）。オレンジの枠はフライト、
    斜線は固体（樹脂の無い場所）。
  </p>
</footer>
</div>
"""

path = os.path.join(OUT, "extruder-fields.html")
with open(path, "w", encoding="utf-8") as fh:
    fh.write(HTML)
print(f"書き出し {path}  {os.path.getsize(path) / 1024:.0f} KB")
print(f"流線 {len(stream_paths)} 本  γ̇max {gam_max:.0f}  μ {mu_min:.0f}–{mu_max:.0f}")
print(f"Q_axial  Newton {fN.Q_axial * 950 * 3600:.1f} kg/h  /  PowerLaw {fP.Q_axial * 950 * 3600:.1f} kg/h")

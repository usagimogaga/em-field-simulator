"""
EM Field Simulator Pro — Streamlit Web Version v2
物理核心完整保留：Biot-Savart、RK4 磁力線、多導線疊加
UI 全面升級：深色科技風、響應式佈局、即時查詢
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import streamlit as st
import math

# ══════════════════════════════════════════════════════════════════
#  頁面設定
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EM Field Simulator Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "EM Field Simulator Pro — 電磁場模擬器"},
)

# ══════════════════════════════════════════════════════════════════
#  全域 CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600&display=swap');

html, body, [class*="css"] {
    background-color: #080810 !important;
    color: #d0d0e8 !important;
    font-family: 'Rajdhani', 'Courier New', monospace;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e0e1a 0%, #0a0a14 100%) !important;
    border-right: 1px solid #1e1e35 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

.app-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    color: #00d4ff;
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0,212,255,0.4);
    margin: 0; line-height: 1.2;
}
.app-sub { font-size: 0.75rem; color: #4a4a80; letter-spacing: 4px; margin-top: 2px; }
.divider { height: 1px; background: linear-gradient(90deg,transparent,#00d4ff55,transparent); margin: 12px 0; }

.sec-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; color: #4a4a80; letter-spacing: 3px;
    text-transform: uppercase; margin: 16px 0 6px;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::after { content:''; flex:1; height:1px; background:#1e1e35; }

.metric-row { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }
.metric-card {
    background:#0f0f1e; border:1px solid #1e1e35;
    border-left:2px solid #ff6b35; border-radius:6px;
    padding:10px 14px; flex:1; min-width:140px;
}
.metric-card.blue  { border-left-color:#00d4ff; }
.metric-card.purple{ border-left-color:#a78bfa; }
.metric-label { font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#5a5a90; letter-spacing:1px; margin-bottom:4px; }
.metric-value { font-family:'Share Tech Mono',monospace; font-size:1.1rem; color:#ff6b35; font-weight:bold; }
.metric-card.blue   .metric-value { color:#00d4ff; }
.metric-card.purple .metric-value { color:#a78bfa; }

.wire-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:#0f0f1e; border:1px solid #1e1e35; border-radius:20px;
    padding:3px 10px; font-family:'Share Tech Mono',monospace;
    font-size:0.7rem; color:#00d4ff; margin:3px 2px;
}
.b-query-box { background:#0f0f1e; border:1px solid #1e1e35; border-radius:8px; padding:14px 18px; }
.b-query-title { font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#4a4a80; letter-spacing:2px; margin-bottom:8px; }
.b-values { display:flex; gap:16px; flex-wrap:wrap; align-items:baseline; }
.b-val-item { display:flex; flex-direction:column; gap:2px; }
.b-val-label { font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#5a5a90; }
.b-val-num  { font-family:'Share Tech Mono',monospace; font-size:1.0rem; color:#a78bfa; }
.b-val-num.total { color:#00d4ff; font-size:1.2rem; }

.plot-wrap { background:#080810; border:1px solid #1a1a30; border-radius:8px; overflow:hidden; }

div[data-testid="stSlider"] label   { color:#5a5a90 !important; font-size:0.75rem !important; }
div[data-testid="stSelectbox"] label{ color:#5a5a90 !important; font-size:0.75rem !important; }
div[data-testid="stCheckbox"] label { color:#a0a0c0 !important; font-size:0.8rem  !important; }
div[data-testid="stNumberInput"] label { color:#5a5a90 !important; font-size:0.72rem !important; }
div[data-testid="stExpander"] { background:#0d0d1a !important; border:1px solid #1a1a30 !important; border-radius:6px !important; }

.stButton > button {
    background:#0d0d1a !important; color:#00d4ff !important;
    border:1px solid #1e1e35 !important; border-radius:5px !important;
    font-family:'Share Tech Mono',monospace !important;
    font-size:0.72rem !important; letter-spacing:1px !important;
    transition:all 0.15s !important; padding:4px 12px !important;
}
.stButton > button:hover { background:#00d4ff22 !important; border-color:#00d4ff !important; }
div[data-testid="stRadio"] label { color:#a0a0c0 !important; font-size:0.82rem !important; }

.legend-row { display:flex; gap:16px; flex-wrap:wrap; margin:6px 0; }
.legend-item { display:flex; align-items:center; gap:5px; font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#5a5a90; }
.legend-dot  { width:8px; height:8px; border-radius:50%; flex-shrink:0; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#080810; }
::-webkit-scrollbar-thumb { background:#1e1e35; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  物理常數
# ══════════════════════════════════════════════════════════════════
MU_0       = 4 * np.pi * 1e-7
WIRE_RAD   = 1e-3
N_WIRE_PTS = 420
N_GRID     = 17
N_3D       = 4

# ══════════════════════════════════════════════════════════════════
#  幾何
# ══════════════════════════════════════════════════════════════════
def wire_straight(L):
    t = np.linspace(-L/2, L/2, N_WIRE_PTS)
    return np.column_stack([np.zeros_like(t), np.zeros_like(t), t])

def wire_loop(R=0.4):
    th = np.linspace(0, 2*np.pi, N_WIRE_PTS)
    return np.column_stack([R*np.cos(th), R*np.sin(th), np.zeros_like(th)])

def wire_solenoid(L, N_turns, R=0.3):
    n_raw = 4000
    th = np.linspace(0, 2*np.pi*N_turns, n_raw)
    z  = np.linspace(-L/2, L/2, n_raw)
    xyz = np.column_stack([R*np.cos(th), R*np.sin(th), z])
    segs = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    arc  = np.concatenate([[0], np.cumsum(segs)])
    s_uni = np.linspace(0, arc[-1], N_WIRE_PTS)
    return np.column_stack([np.interp(s_uni, arc, xyz[:,i]) for i in range(3)])

def apply_transform(wire, pos, rot_deg):
    rx, ry, rz = [np.radians(a) for a in rot_deg]
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return wire @ (Rz @ Ry @ Rx).T + np.array(pos)

def build_wire(wd):
    if   wd["mode"] == "Straight": w = wire_straight(wd["L"])
    elif wd["mode"] == "Loop":     w = wire_loop()
    else:                          w = wire_solenoid(wd["L"], wd["N"])
    return apply_transform(w, wd.get("pos",[0,0,0]), wd.get("rot",[0,0,0]))

# ══════════════════════════════════════════════════════════════════
#  Biot-Savart
# ══════════════════════════════════════════════════════════════════
def biot_savart(wire, I, obs):
    dl  = np.diff(wire, axis=0).astype(np.float64)
    mid = (wire[:-1] + dl * 0.5).astype(np.float64)
    prefactor = MU_0 * I / (4.0 * np.pi)
    seg_len   = np.linalg.norm(dl, axis=1).mean()
    wire_rad2 = float(np.clip(seg_len * 0.3, WIRE_RAD, 5*WIRE_RAD) ** 2)
    shape    = obs.shape[:-1]
    obs_flat = obs.reshape(-1, 3).astype(np.float64)
    r     = obs_flat[:, np.newaxis, :] - mid
    r_mag = np.linalg.norm(r, axis=-1)
    denom = (r_mag**2 + wire_rad2)**1.5
    dB    = prefactor * np.cross(dl, r) / denom[..., np.newaxis]
    return dB.sum(axis=1).reshape(shape + (3,))

# ══════════════════════════════════════════════════════════════════
#  RK4 磁力線
# ══════════════════════════════════════════════════════════════════
def trace_field_line(wires_I, seed, ds=0.02, n_steps=200):
    def B_at(p):
        return sum(biot_savart(w, I, p.reshape(1,3)) for w, I in wires_I)[0]
    pts = [seed.copy()]; p = seed.copy()
    for _ in range(n_steps):
        b1 = B_at(p); n1 = np.linalg.norm(b1)
        if n1 < 1e-20: break
        k1 = b1/n1
        b2 = B_at(p+0.5*ds*k1); n2 = np.linalg.norm(b2)
        if n2 < 1e-20: break
        k2 = b2/n2
        b3 = B_at(p+0.5*ds*k2); n3 = np.linalg.norm(b3)
        if n3 < 1e-20: break
        k3 = b3/n3
        b4 = B_at(p+ds*k3); n4 = np.linalg.norm(b4)
        if n4 < 1e-20: break
        k4 = b4/n4
        p  = p + (ds/6)*(k1+2*k2+2*k3+k4)
        if np.linalg.norm(p) > 1.5: break
        pts.append(p.copy())
    return np.array(pts)

# ══════════════════════════════════════════════════════════════════
#  解析估算
# ══════════════════════════════════════════════════════════════════
def analytical_values(mode, I, L, N, freq=0.0, ac=False):
    vals = {}
    if mode == "Straight":
        r = 0.05
        vals["B (r=5cm)"] = (MU_0*I/(2*np.pi*r)*1e6, "µT")
        vals["F / len"]   = (MU_0*I**2/(2*np.pi*r)*1e6, "µN/m")
    elif mode == "Loop":
        R=0.4; a=1e-3
        Li = MU_0*R*(np.log(8*R/a)-2)
        vals["B 中心"]    = (MU_0*I/(2*R)*1e6, "µT")
        vals["電感 L"]    = (Li*1e6, "µH")
        vals["能量 ½LI²"] = (0.5*Li*I**2*1e9, "nJ")
        if ac and freq > 0:
            vals["峰值 EMF"] = (Li*I*2*np.pi*freq*1e3, "mV")
    else:
        n=N/L; R=0.3
        Li = MU_0*n**2*np.pi*R**2*L
        vals["B 中心"]    = (MU_0*n*I*1e3, "mT")
        vals["電感 L"]    = (Li*1e3, "mH")
        vals["能量 ½LI²"] = (0.5*Li*I**2*1e6, "µJ")
        vals["線圈密度 n"] = (n, "turns/m")
        if ac and freq > 0:
            vals["峰值 EMF"] = (Li*I*2*np.pi*freq*1e3, "mV")
    return vals

# ══════════════════════════════════════════════════════════════════
#  磁力線種子
# ══════════════════════════════════════════════════════════════════
def fl_seeds_2d(plane, mode):
    if mode == "Loop":
        if plane == "xy":
            a = np.linspace(0, 2*np.pi, 8, endpoint=False)
            return [(0.42*np.cos(t), 0.42*np.sin(t)) for t in a]
        return ([(0.42,z) for z in np.linspace(-0.6,0.6,5)] +
                [(-0.42,z) for z in np.linspace(-0.6,0.6,5)] +
                [(0.02,z) for z in np.linspace(-0.5,0.5,4)])
    elif mode == "Solenoid":
        if plane == "xy": return []
        interior = [(x,z) for x in [-0.15,0.0,0.15] for z in np.linspace(-0.42,0.42,4)]
        exterior = ([(0.38,z) for z in np.linspace(-0.55,0.55,5)] +
                    [(-0.38,z) for z in np.linspace(-0.55,0.55,5)])
        return interior + exterior
    else:
        if plane == "xy":
            return [(r,0.0) for r in [0.08,0.18,0.32,0.50]]
        return []

# ══════════════════════════════════════════════════════════════════
#  繪圖
# ══════════════════════════════════════════════════════════════════
BG = "#080810"; ACCENT = "#00d4ff"; ACCENT2 = "#ff6b35"; ACCENT3 = "#a78bfa"

def make_2d_fig(U, V, Bu, Bv, wires_I, plane, mode, show_fl, cmap):
    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    B_mag = np.sqrt(Bu**2 + Bv**2) + 1e-30
    cf = ax.contourf(U, V, B_mag, levels=20, cmap=cmap, alpha=0.88)
    log_s = np.log1p(B_mag / B_mag.max())
    ax.quiver(U, V, Bu/B_mag*log_s, Bv/B_mag*log_s,
              color="white", scale=4.0, width=0.004, alpha=0.55,
              headwidth=4, headlength=5)
    if show_fl:
        for sx, sy in fl_seeds_2d(plane, mode):
            if   plane=="xy": s3=np.array([sx,sy,0.0])
            elif plane=="xz": s3=np.array([sx,0.0,sy])
            else:             s3=np.array([0.0,sx,sy])
            fl = trace_field_line(wires_I, s3, ds=0.02, n_steps=200)
            if len(fl) < 2: continue
            if   plane=="xy": px,py=fl[:,0],fl[:,1]
            elif plane=="xz": px,py=fl[:,0],fl[:,2]
            else:             px,py=fl[:,1],fl[:,2]
            mask = (np.abs(px)<0.72)&(np.abs(py)<0.72)
            if mask.sum() > 1:
                ax.plot(px[mask], py[mask], color=ACCENT, lw=1.0, alpha=0.75,
                        path_effects=[pe.Stroke(linewidth=2.2, foreground="#00d4ff1a"), pe.Normal()])
    for i, (w, _) in enumerate(wires_I):
        if   plane=="xy": pts=w[np.abs(w[:,2])<0.05]; xs,ys=pts[:,0],pts[:,1]
        elif plane=="xz": pts=w[np.abs(w[:,1])<0.05]; xs,ys=pts[:,0],pts[:,2]
        else:             pts=w[np.abs(w[:,0])<0.05]; xs,ys=pts[:,1],pts[:,2]
        if len(xs):
            c = ACCENT2 if i==0 else ACCENT3
            ax.scatter(xs, ys, c=c, s=22, zorder=6, linewidths=0.8, edgecolors="#ffffff33")
    labels = {"xy":("X [m]","Y [m]","XY  z=0"),
              "xz":("X [m]","Z [m]","XZ  y=0"),
              "yz":("Y [m]","Z [m]","YZ  x=0")}
    xl,yl,title = labels[plane]
    ax.set_xlim(-0.7,0.7); ax.set_ylim(-0.7,0.7)
    ax.set_xlabel(xl, color="#3a3a60", fontsize=7, labelpad=2)
    ax.set_ylabel(yl, color="#3a3a60", fontsize=7, labelpad=2)
    ax.set_title(title, color=ACCENT, fontsize=9, fontfamily="monospace", pad=5, fontweight="bold")
    ax.tick_params(colors="#3a3a60", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1a30"); sp.set_linewidth(0.5)
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.03)
    cbar.ax.tick_params(colors="#5a5a90", labelsize=6)
    cbar.outline.set_edgecolor("#1a1a30")
    cbar.set_label("|B| [T]", color="#5a5a90", fontsize=6)
    fig.tight_layout(pad=0.4)
    return fig

def make_3d_fig(wires_I, mode, show_fl):
    fig = plt.figure(figsize=(4.6, 4.2), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    l3 = np.linspace(-0.6, 0.6, N_3D)
    X,Y,Z = np.meshgrid(l3,l3,l3)
    B3d = sum(biot_savart(w,I,np.stack([X,Y,Z],-1)) for w,I in wires_I)
    Bx,By,Bz = B3d[...,0],B3d[...,1],B3d[...,2]
    Bm = np.sqrt(Bx**2+By**2+Bz**2)+1e-30
    ax.quiver(X,Y,Z, Bx/Bm, By/Bm, Bz/Bm, length=0.14, linewidth=0.8,
              color=plt.cm.cool(Bm.ravel()/Bm.max()), alpha=0.75)
    for i,(w,_) in enumerate(wires_I):
        ax.plot(w[:,0],w[:,1],w[:,2], color=ACCENT2 if i==0 else ACCENT3, lw=2.0, alpha=0.92)
    if show_fl:
        seeds = []
        if mode=="Loop":
            seeds=[np.array([0.42*np.cos(t),0.42*np.sin(t),0.0]) for t in np.linspace(0,2*np.pi,6,endpoint=False)]
        elif mode=="Solenoid":
            for ang in np.linspace(0,2*np.pi,4,endpoint=False):
                seeds.append(np.array([0.32*np.cos(ang),0.32*np.sin(ang),0.0]))
            for z in np.linspace(-0.28,0.28,3):
                seeds.append(np.array([0.01,0.0,z]))
        else:
            seeds=[np.array([r,0.0,0.0]) for r in [0.07,0.15,0.28,0.44]]
        for seed in seeds:
            fl=trace_field_line(wires_I,seed,ds=0.025,n_steps=130)
            if len(fl)>2: ax.plot(fl[:,0],fl[:,1],fl[:,2],color=ACCENT,lw=0.9,alpha=0.55)
    ax.set_title("3D", color=ACCENT, fontsize=9, fontfamily="monospace", pad=2, fontweight="bold")
    ax.set_axis_off()
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("#0d0d20")
    fig.tight_layout(pad=0.2)
    return fig

# ══════════════════════════════════════════════════════════════════
#  Session State
# ══════════════════════════════════════════════════════════════════
if "wire_list" not in st.session_state:
    st.session_state.wire_list = [
        {"mode":"Solenoid","I":40,"L":1.0,"N":12,"pos":[0.0,0.0,0.0],"rot":[0.0,0.0,0.0]}
    ]

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="app-title">⚡ EM FIELD</div><div class="app-sub">SIMULATOR  PRO</div><div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">幾何形狀</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Straight","Loop","Solenoid"], index=2, horizontal=True, label_visibility="collapsed")

    st.markdown('<div class="sec-label">參數</div>', unsafe_allow_html=True)
    I_val = st.slider("電流  I  (A)", 1.0, 100.0, 40.0, 1.0)
    L_val = st.slider("長度  L  (m)", 0.2, 2.0,   1.0,  0.05)
    N_val = st.slider("匝數  N",      1,   40,    12,   1)

    st.markdown('<div class="sec-label">交流模式</div>', unsafe_allow_html=True)
    ac_mode = st.checkbox("⚡ 啟用 AC", value=False)
    freq    = st.slider("頻率 (Hz)", 0, 120, 50, 5, disabled=not ac_mode)

    st.markdown('<div class="sec-label">顯示</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    show_fl  = c1.checkbox("磁力線", value=True)
    cmap_opt = c2.selectbox("色彩", ["magma","plasma","viridis","inferno","hot"], label_visibility="collapsed")

    st.markdown('<div class="sec-label">多導線</div>', unsafe_allow_html=True)
    for i, wd in enumerate(st.session_state.wire_list):
        ca, cb = st.columns([5,1])
        with ca:
            clr = "#00d4ff" if i==0 else "#a78bfa"
            st.markdown(f'<span class="wire-pill" style="color:{clr}">#{i+1} {wd["mode"][:3].upper()}  I={wd["I"]:.0f}A</span>', unsafe_allow_html=True)
        with cb:
            if i>0 and st.button("✕", key=f"rm_{i}"):
                st.session_state.wire_list.pop(i); st.rerun()

    with st.expander("＋ 新增導線"):
        nw_mode = st.selectbox("類型", ["Straight","Loop","Solenoid"], key="nw_m")
        nw_I = st.number_input("電流 (A)", 1.0, 200.0, 20.0, key="nw_I")
        nw_L = st.number_input("長度 (m)", 0.1, 3.0,   1.0,  key="nw_L")
        nw_N = st.number_input("匝數",     1,   60,    8,    key="nw_N", step=1)
        st.caption("📍 位置 (x, y, z) m")
        p1,p2,p3 = st.columns(3)
        nw_px=p1.number_input("X",-1.5,1.5,0.0,0.1,key="px")
        nw_py=p2.number_input("Y",-1.5,1.5,0.0,0.1,key="py")
        nw_pz=p3.number_input("Z",-1.5,1.5,0.0,0.1,key="pz")
        st.caption("🔄 旋轉 (rx, ry, rz) °")
        r1,r2,r3 = st.columns(3)
        nw_rx=r1.number_input("Rx",-180.0,180.0,0.0, 5.0,key="rx")
        nw_ry=r2.number_input("Ry",-180.0,180.0,0.0, 5.0,key="ry")
        nw_rz=r3.number_input("Rz",-180.0,180.0,90.0,5.0,key="rz")
        if st.button("✓ 加入", use_container_width=True):
            st.session_state.wire_list.append({
                "mode":str(nw_mode),"I":float(nw_I),"L":float(nw_L),"N":int(nw_N),
                "pos":[nw_px,nw_py,nw_pz],"rot":[nw_rx,nw_ry,nw_rz],
            }); st.rerun()

    if len(st.session_state.wire_list)>1:
        if st.button("✕ 清除副導線", use_container_width=True):
            st.session_state.wire_list=[st.session_state.wire_list[0]]; st.rerun()

# ══════════════════════════════════════════════════════════════════
#  主視窗
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px">
  <span style="font-family:'Share Tech Mono',monospace;font-size:1.5rem;color:#00d4ff;
               letter-spacing:3px;text-shadow:0 0 16px rgba(0,212,255,0.35)">
    ⚡ EM FIELD SIMULATOR PRO
  </span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:#2a2a50;letter-spacing:2px">
    BIOT–SAVART · RK4 FIELD LINES · MULTI-WIRE
  </span>
</div>
<div style="height:1px;background:linear-gradient(90deg,#00d4ff55,transparent);margin-bottom:14px"></div>
""", unsafe_allow_html=True)

# 同步主導線
st.session_state.wire_list[0].update({"mode":mode,"I":I_val,"L":L_val,"N":N_val})

# 建立 wires_I
wires_I = []
for wd in st.session_state.wire_list:
    w = build_wire(wd)
    I = wd["I"] * (math.cos(0.0) if (ac_mode and freq>0) else 1.0)
    wires_I.append((w, I))

# 計算場
with st.spinner("⚡  計算電磁場…"):
    l   = np.linspace(-0.7, 0.7, N_GRID)
    U,V = np.meshgrid(l, l)
    z   = np.zeros_like(U)
    def sumB(obs):
        return sum(biot_savart(w,I,obs) for w,I in wires_I)
    B_xy = sumB(np.stack([U,V,z],-1))
    B_xz = sumB(np.stack([U,z,V],-1))
    B_yz = sumB(np.stack([z,U,V],-1))

# 四格圖
col1,col2 = st.columns(2, gap="small")
with col1:
    st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
    st.pyplot(make_3d_fig(wires_I,mode,show_fl), use_container_width=True)
    st.markdown('</div><div class="plot-wrap" style="margin-top:8px">', unsafe_allow_html=True)
    st.pyplot(make_2d_fig(U,V,B_xy[...,0],B_xy[...,1],wires_I,"xy",mode,show_fl,cmap_opt), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
    st.pyplot(make_2d_fig(U,V,B_xz[...,0],B_xz[...,2],wires_I,"xz",mode,show_fl,cmap_opt), use_container_width=True)
    st.markdown('</div><div class="plot-wrap" style="margin-top:8px">', unsafe_allow_html=True)
    st.pyplot(make_2d_fig(U,V,B_yz[...,1],B_yz[...,2],wires_I,"yz",mode,show_fl,cmap_opt), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="legend-row" style="margin-top:6px">
  <div class="legend-item"><div class="legend-dot" style="background:#ff6b35"></div>主導線截面</div>
  <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div>副導線截面</div>
  <div class="legend-item"><div class="legend-dot" style="background:#00d4ff"></div>磁力線 (RK4)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ffffff88"></div>場方向箭頭</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  解析估算
# ══════════════════════════════════════════════════════════════════
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#1e1e35,transparent);margin:16px 0 10px"></div><div class="sec-label">解析估算</div>', unsafe_allow_html=True)
nums = analytical_values(mode, I_val, L_val, N_val, freq=freq, ac=ac_mode)
palette = ["blue","blue","purple","purple","blue","purple"]
cards_html = '<div class="metric-row">'
for (k,(v,unit)), c in zip(nums.items(), palette):
    cards_html += f'<div class="metric-card {c}"><div class="metric-label">{k}</div><div class="metric-value">{v:.4g} <span style="font-size:0.7rem;color:#5a5a90">{unit}</span></div></div>'
cards_html += '</div>'
st.markdown(cards_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  任意點 B 場查詢
# ══════════════════════════════════════════════════════════════════
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#1e1e35,transparent);margin:16px 0 10px"></div><div class="sec-label">任意點磁場查詢</div>', unsafe_allow_html=True)
qc1,qc2,qc3,qc4 = st.columns([1,1,1,3])
qx=qc1.number_input("X (m)",-0.7,0.7,0.0,0.05,key="qx",format="%.2f")
qy=qc2.number_input("Y (m)",-0.7,0.7,0.0,0.05,key="qy",format="%.2f")
qz=qc3.number_input("Z (m)",-0.7,0.7,0.0,0.05,key="qz",format="%.2f")
B_pt  = sum(biot_savart(w,I,np.array([[qx,qy,qz]])) for w,I in wires_I)[0]*1e6
Bm_pt = np.linalg.norm(B_pt)
with qc4:
    st.markdown(f"""
<div class="b-query-box">
  <div class="b-query-title">B  FIELD  AT  ({qx:.2f}, {qy:.2f}, {qz:.2f}) m</div>
  <div class="b-values">
    <div class="b-val-item"><span class="b-val-label">Bx [µT]</span><span class="b-val-num">{B_pt[0]:+.4f}</span></div>
    <div class="b-val-item"><span class="b-val-label">By [µT]</span><span class="b-val-num">{B_pt[1]:+.4f}</span></div>
    <div class="b-val-item"><span class="b-val-label">Bz [µT]</span><span class="b-val-num">{B_pt[2]:+.4f}</span></div>
    <div class="b-val-item"><span class="b-val-label">|B| [µT]</span><span class="b-val-num total">{Bm_pt:.4f}</span></div>
  </div>
</div>""", unsafe_allow_html=True)

# Footer
st.markdown('<div style="margin-top:32px;padding-top:12px;border-top:1px solid #1a1a30;text-align:center;font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;color:#2a2a50;letter-spacing:2px">EM FIELD SIMULATOR PRO  ·  BIOT–SAVART LAW  ·  RK4 FIELD LINE TRACER</div>', unsafe_allow_html=True)

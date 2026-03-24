"""
EM Field Simulator Pro — Streamlit Full Version v3
Physics  : Biot-Savart, RK4 field lines, multi-wire superposition, AC animation
UI       : 8-language, interactive Plotly 3D (drag/rotate/hover),
           scene save/load JSON, CSV export, PNG screenshot download
Deploy   : Render (private repo) or Streamlit Cloud
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import plotly.graph_objects as go
import streamlit as st
import math, json, csv, io, datetime

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EM Field Simulator Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  I18N  — 8 languages
# ══════════════════════════════════════════════════════════════════
LANGS = {
"繁中": dict(
    flag="🇹🇼", title="電磁場模擬器",
    subtitle="Biot–Savart · RK4 磁力線 · 多導線疊加",
    geometry="幾何形狀", parameters="參數", display="顯示設定",
    wires="多導線管理", ac_sec="交流模式", io_sec="場景 I/O",
    straight="直導線", loop="圓形線圈", solenoid="螺線管",
    current="電流  I (A)", length="長度  L (m)", turns="匝數  N",
    freq="頻率 (Hz)", ac_on="⚡ 啟用 AC 動畫",
    show_fl="磁力線", cmap_lbl="色彩",
    add_wire="＋ 新增導線", clear_extra="✕ 清除副導線",
    w_type="類型", w_I="電流 (A)", w_L="長度 (m)", w_N="匝數",
    w_pos="位置 (x,y,z) m", w_rot="旋轉 (rx,ry,rz) °",
    w_add="✓ 加入", w_del="✕",
    save_json="💾 儲存場景", load_json="📂 載入場景",
    export_csv="⬇ 匯出 CSV", screenshot="📷 下載截圖",
    analytics="解析估算", query="任意點 B 場查詢",
    b_ctr="B 中心", induc="電感 L", energy="能量 ½LI²",
    n_dens="線圈密度 n", emf="峰值 EMF",
    b_r5="B (r=5cm)", force="F / 長度",
    ac_note="AC：I = I·cos(2πft)，每次更新 t 遞增",
    footer="電磁場模擬器 Pro  ·  Biot-Savart 定律  ·  RK4 磁力線追蹤",
    plane_xy="XY 平面  z=0", plane_xz="XZ 平面  y=0", plane_yz="YZ 平面  x=0",
    p3d="3D 向量場（可拖曳旋轉）", bq_title="B 場",
    ac_live="⚡ AC 即時", lang_lbl="語言",
),
"EN": dict(
    flag="🇺🇸", title="EM Field Simulator",
    subtitle="Biot–Savart · RK4 Field Lines · Multi-Wire",
    geometry="Geometry", parameters="Parameters", display="Display",
    wires="Wire Manager", ac_sec="AC Mode", io_sec="Scene I/O",
    straight="Straight", loop="Loop", solenoid="Solenoid",
    current="Current  I (A)", length="Length  L (m)", turns="Turns  N",
    freq="Frequency (Hz)", ac_on="⚡ Enable AC Animation",
    show_fl="Field Lines", cmap_lbl="Colormap",
    add_wire="＋ Add Wire", clear_extra="✕ Clear Extra",
    w_type="Type", w_I="Current (A)", w_L="Length (m)", w_N="Turns",
    w_pos="Position (x,y,z) m", w_rot="Rotation (rx,ry,rz) °",
    w_add="✓ Add", w_del="✕",
    save_json="💾 Save Scene", load_json="📂 Load Scene",
    export_csv="⬇ Export CSV", screenshot="📷 Download Screenshot",
    analytics="Analytical Estimates", query="B-Field Point Query",
    b_ctr="B centre", induc="Inductance L", energy="Energy ½LI²",
    n_dens="Turn density n", emf="Peak EMF",
    b_r5="B (r=5cm)", force="F / length",
    ac_note="AC: I = I·cos(2πft), t increments each refresh",
    footer="EM Field Simulator Pro  ·  Biot-Savart Law  ·  RK4 Field Line Tracer",
    plane_xy="XY Plane  z=0", plane_xz="XZ Plane  y=0", plane_yz="YZ Plane  x=0",
    p3d="3D Field Vectors (drag to rotate)", bq_title="B Field",
    ac_live="⚡ AC LIVE", lang_lbl="Language",
),
"日本語": dict(
    flag="🇯🇵", title="電磁場シミュレーター",
    subtitle="Biot–Savart · RK4 磁力線 · 複数導線",
    geometry="形状", parameters="パラメータ", display="表示設定",
    wires="導線管理", ac_sec="交流モード", io_sec="シーン I/O",
    straight="直線導線", loop="円形コイル", solenoid="ソレノイド",
    current="電流  I (A)", length="長さ  L (m)", turns="巻数  N",
    freq="周波数 (Hz)", ac_on="⚡ AC アニメーション",
    show_fl="磁力線", cmap_lbl="カラーマップ",
    add_wire="＋ 導線を追加", clear_extra="✕ 副導線を削除",
    w_type="種類", w_I="電流 (A)", w_L="長さ (m)", w_N="巻数",
    w_pos="位置 (x,y,z) m", w_rot="回転 (rx,ry,rz) °",
    w_add="✓ 追加", w_del="✕",
    save_json="💾 シーン保存", load_json="📂 シーン読込",
    export_csv="⬇ CSV エクスポート", screenshot="📷 スクリーンショット",
    analytics="解析的推定値", query="任意点 B 場クエリ",
    b_ctr="中心 B", induc="インダクタンス L", energy="エネルギー ½LI²",
    n_dens="線密度 n", emf="ピーク EMF",
    b_r5="B (r=5cm)", force="F / 長さ",
    ac_note="AC モード：I = I·cos(2πft)",
    footer="電磁場シミュレーター Pro  ·  Biot-Savart 法則  ·  RK4 磁力線",
    plane_xy="XY 平面  z=0", plane_xz="XZ 平面  y=0", plane_yz="YZ 平面  x=0",
    p3d="3D ベクトル場（ドラッグで回転）", bq_title="B 場",
    ac_live="⚡ AC ライブ", lang_lbl="言語",
),
"简中": dict(
    flag="🇨🇳", title="电磁场模拟器",
    subtitle="Biot–Savart · RK4 磁力线 · 多导线叠加",
    geometry="几何形状", parameters="参数", display="显示设置",
    wires="多导线管理", ac_sec="交流模式", io_sec="场景 I/O",
    straight="直导线", loop="圆形线圈", solenoid="螺线管",
    current="电流  I (A)", length="长度  L (m)", turns="匝数  N",
    freq="频率 (Hz)", ac_on="⚡ 启用 AC 动画",
    show_fl="磁力线", cmap_lbl="色彩",
    add_wire="＋ 新增导线", clear_extra="✕ 清除副导线",
    w_type="类型", w_I="电流 (A)", w_L="长度 (m)", w_N="匝数",
    w_pos="位置 (x,y,z) m", w_rot="旋转 (rx,ry,rz) °",
    w_add="✓ 加入", w_del="✕",
    save_json="💾 保存场景", load_json="📂 载入场景",
    export_csv="⬇ 导出 CSV", screenshot="📷 下载截图",
    analytics="解析估算", query="任意点 B 场查询",
    b_ctr="B 中心", induc="电感 L", energy="能量 ½LI²",
    n_dens="线圈密度 n", emf="峰值 EMF",
    b_r5="B (r=5cm)", force="F / 长度",
    ac_note="AC：I = I·cos(2πft)，每次刷新 t 递增",
    footer="电磁场模拟器 Pro  ·  Biot-Savart 定律  ·  RK4 磁力线追踪",
    plane_xy="XY 平面  z=0", plane_xz="XZ 平面  y=0", plane_yz="YZ 平面  x=0",
    p3d="3D 向量场（可拖动旋转）", bq_title="B 场",
    ac_live="⚡ AC 实时", lang_lbl="语言",
),
"Español": dict(
    flag="🇪🇸", title="Simulador de Campo EM",
    subtitle="Biot–Savart · Líneas RK4 · Múltiples Cables",
    geometry="Geometría", parameters="Parámetros", display="Visualización",
    wires="Gestor de Cables", ac_sec="Modo AC", io_sec="Escena I/O",
    straight="Recto", loop="Bucle", solenoid="Solenoide",
    current="Corriente  I (A)", length="Longitud  L (m)", turns="Vueltas  N",
    freq="Frecuencia (Hz)", ac_on="⚡ Activar Animación AC",
    show_fl="Líneas de Campo", cmap_lbl="Mapa de Color",
    add_wire="＋ Añadir Cable", clear_extra="✕ Eliminar Extras",
    w_type="Tipo", w_I="Corriente (A)", w_L="Longitud (m)", w_N="Vueltas",
    w_pos="Posición (x,y,z) m", w_rot="Rotación (rx,ry,rz) °",
    w_add="✓ Añadir", w_del="✕",
    save_json="💾 Guardar Escena", load_json="📂 Cargar Escena",
    export_csv="⬇ Exportar CSV", screenshot="📷 Descargar Captura",
    analytics="Estimaciones Analíticas", query="Consulta B en Punto",
    b_ctr="B centro", induc="Inductancia L", energy="Energía ½LI²",
    n_dens="Densidad n", emf="EMF Pico",
    b_r5="B (r=5cm)", force="F / longitud",
    ac_note="AC: I = I·cos(2πft)",
    footer="Simulador EM Pro  ·  Ley de Biot-Savart  ·  Líneas de Campo RK4",
    plane_xy="Plano XY  z=0", plane_xz="Plano XZ  y=0", plane_yz="Plano YZ  x=0",
    p3d="Vectores 3D (arrastrar para rotar)", bq_title="Campo B",
    ac_live="⚡ AC VIVO", lang_lbl="Idioma",
),
"Deutsch": dict(
    flag="🇩🇪", title="EM-Feldsimulator",
    subtitle="Biot–Savart · RK4 Feldlinien · Mehrere Leiter",
    geometry="Geometrie", parameters="Parameter", display="Anzeige",
    wires="Leiterverwaltung", ac_sec="Wechselstrom", io_sec="Szene I/O",
    straight="Gerade", loop="Schleife", solenoid="Solenoid",
    current="Strom  I (A)", length="Länge  L (m)", turns="Windungen  N",
    freq="Frequenz (Hz)", ac_on="⚡ AC-Animation aktivieren",
    show_fl="Feldlinien", cmap_lbl="Farbskala",
    add_wire="＋ Leiter hinzufügen", clear_extra="✕ Extras löschen",
    w_type="Typ", w_I="Strom (A)", w_L="Länge (m)", w_N="Windungen",
    w_pos="Position (x,y,z) m", w_rot="Rotation (rx,ry,rz) °",
    w_add="✓ Hinzufügen", w_del="✕",
    save_json="💾 Szene speichern", load_json="📂 Szene laden",
    export_csv="⬇ CSV exportieren", screenshot="📷 Screenshot",
    analytics="Analytische Schätzungen", query="B-Feld Abfrage",
    b_ctr="B Zentrum", induc="Induktivität L", energy="Energie ½LI²",
    n_dens="Windungsdichte n", emf="Spitzen-EMK",
    b_r5="B (r=5cm)", force="F / Länge",
    ac_note="AC: I = I·cos(2πft)",
    footer="EM-Feldsimulator Pro  ·  Biot-Savart-Gesetz  ·  RK4 Feldlinien",
    plane_xy="XY-Ebene  z=0", plane_xz="XZ-Ebene  y=0", plane_yz="YZ-Ebene  x=0",
    p3d="3D-Vektoren (ziehen zum Drehen)", bq_title="B-Feld",
    ac_live="⚡ AC LIVE", lang_lbl="Sprache",
),
"Français": dict(
    flag="🇫🇷", title="Simulateur de Champ EM",
    subtitle="Biot–Savart · Lignes RK4 · Multi-fils",
    geometry="Géométrie", parameters="Paramètres", display="Affichage",
    wires="Gestion des fils", ac_sec="Mode AC", io_sec="Scène I/O",
    straight="Droit", loop="Boucle", solenoid="Solénoïde",
    current="Courant  I (A)", length="Longueur  L (m)", turns="Spires  N",
    freq="Fréquence (Hz)", ac_on="⚡ Activer Animation AC",
    show_fl="Lignes de champ", cmap_lbl="Palette",
    add_wire="＋ Ajouter fil", clear_extra="✕ Effacer extras",
    w_type="Type", w_I="Courant (A)", w_L="Longueur (m)", w_N="Spires",
    w_pos="Position (x,y,z) m", w_rot="Rotation (rx,ry,rz) °",
    w_add="✓ Ajouter", w_del="✕",
    save_json="💾 Sauvegarder", load_json="📂 Charger",
    export_csv="⬇ Exporter CSV", screenshot="📷 Capturer",
    analytics="Estimations analytiques", query="Requête B en point",
    b_ctr="B centre", induc="Inductance L", energy="Énergie ½LI²",
    n_dens="Densité n", emf="FEM Crête",
    b_r5="B (r=5cm)", force="F / longueur",
    ac_note="AC : I = I·cos(2πft)",
    footer="Simulateur EM Pro  ·  Loi de Biot-Savart  ·  Lignes RK4",
    plane_xy="Plan XY  z=0", plane_xz="Plan XZ  y=0", plane_yz="Plan YZ  x=0",
    p3d="Vecteurs 3D (glisser pour tourner)", bq_title="Champ B",
    ac_live="⚡ AC EN DIRECT", lang_lbl="Langue",
),
"한국어": dict(
    flag="🇰🇷", title="전자기장 시뮬레이터",
    subtitle="Biot–Savart · RK4 자기력선 · 다중 도선",
    geometry="형상", parameters="매개변수", display="표시 설정",
    wires="도선 관리", ac_sec="교류 모드", io_sec="장면 I/O",
    straight="직선 도선", loop="원형 코일", solenoid="솔레노이드",
    current="전류  I (A)", length="길이  L (m)", turns="권수  N",
    freq="주파수 (Hz)", ac_on="⚡ AC 애니메이션 활성화",
    show_fl="자기력선", cmap_lbl="색상표",
    add_wire="＋ 도선 추가", clear_extra="✕ 부가 도선 삭제",
    w_type="유형", w_I="전류 (A)", w_L="길이 (m)", w_N="권수",
    w_pos="위치 (x,y,z) m", w_rot="회전 (rx,ry,rz) °",
    w_add="✓ 추가", w_del="✕",
    save_json="💾 장면 저장", load_json="📂 장면 불러오기",
    export_csv="⬇ CSV 내보내기", screenshot="📷 스크린샷",
    analytics="해석적 추정", query="임의 점 B 장 조회",
    b_ctr="B 중심", induc="인덕턴스 L", energy="에너지 ½LI²",
    n_dens="권선 밀도 n", emf="피크 EMF",
    b_r5="B (r=5cm)", force="F / 길이",
    ac_note="AC: I = I·cos(2πft)",
    footer="전자기장 시뮬레이터 Pro  ·  Biot-Savart 법칙  ·  RK4 자기력선",
    plane_xy="XY 평면  z=0", plane_xz="XZ 평면  y=0", plane_yz="YZ 평면  x=0",
    p3d="3D 벡터장 (드래그하여 회전)", bq_title="B 장",
    ac_live="⚡ AC 실시간", lang_lbl="언어",
),
}

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600&display=swap');
html,body,[class*="css"]{background:#080810!important;color:#d0d0e8!important;
  font-family:'Rajdhani','Courier New',monospace;}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0e0e1a,#0a0a14)!important;
  border-right:1px solid #1e1e35!important;}
.app-title{font-family:'Share Tech Mono',monospace;font-size:1.6rem;color:#00d4ff;
  letter-spacing:3px;text-shadow:0 0 18px rgba(0,212,255,.4);line-height:1.2;}
.app-sub{font-size:.7rem;color:#4a4a80;letter-spacing:3px;margin-top:2px;}
.hdiv{height:1px;background:linear-gradient(90deg,transparent,#00d4ff44,transparent);margin:10px 0;}
.slbl{font-family:'Share Tech Mono',monospace;font-size:.6rem;color:#4a4a80;
  letter-spacing:3px;text-transform:uppercase;margin:12px 0 4px;
  display:flex;align-items:center;gap:8px;}
.slbl::after{content:'';flex:1;height:1px;background:#1e1e35;}
.mr{display:flex;flex-wrap:wrap;gap:7px;margin:5px 0;}
.mc{background:#0f0f1e;border:1px solid #1e1e35;border-left:2px solid #ff6b35;
  border-radius:6px;padding:8px 12px;flex:1;min-width:120px;}
.mc.b{border-left-color:#00d4ff;}.mc.p{border-left-color:#a78bfa;}
.ml{font-family:'Share Tech Mono',monospace;font-size:.6rem;color:#5a5a90;
  letter-spacing:1px;margin-bottom:3px;}
.mv{font-family:'Share Tech Mono',monospace;font-size:1rem;color:#ff6b35;font-weight:bold;}
.mc.b .mv{color:#00d4ff;}.mc.p .mv{color:#a78bfa;}
.wpill{display:inline-flex;align-items:center;gap:5px;background:#0f0f1e;
  border:1px solid #1e1e35;border-radius:20px;padding:2px 9px;
  font-family:'Share Tech Mono',monospace;font-size:.67rem;color:#00d4ff;margin:2px;}
.bqbox{background:#0f0f1e;border:1px solid #1e1e35;border-radius:8px;padding:12px 16px;}
.bqtitle{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:#4a4a80;
  letter-spacing:2px;margin-bottom:6px;}
.bvals{display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;}
.bvi{display:flex;flex-direction:column;gap:1px;}
.bvl{font-family:'Share Tech Mono',monospace;font-size:.56rem;color:#5a5a90;}
.bvn{font-family:'Share Tech Mono',monospace;font-size:.92rem;color:#a78bfa;}
.bvn.tot{color:#00d4ff;font-size:1.1rem;}
.pw{background:#080810;border:1px solid #1a1a30;border-radius:8px;overflow:hidden;}
.lr{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0;}
.li{display:flex;align-items:center;gap:4px;font-family:'Share Tech Mono',monospace;
  font-size:.6rem;color:#5a5a90;}
.ld{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
div[data-testid="stSlider"] label,div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label{color:#5a5a90!important;font-size:.72rem!important;}
div[data-testid="stCheckbox"] label{color:#a0a0c0!important;font-size:.78rem!important;}
div[data-testid="stExpander"]{background:#0d0d1a!important;border:1px solid #1a1a30!important;
  border-radius:6px!important;}
div[data-testid="stRadio"] label{color:#a0a0c0!important;font-size:.78rem!important;}
.stButton>button{background:#0d0d1a!important;color:#00d4ff!important;
  border:1px solid #1e1e35!important;border-radius:5px!important;
  font-family:'Share Tech Mono',monospace!important;font-size:.68rem!important;
  letter-spacing:1px!important;padding:3px 9px!important;transition:all .15s!important;}
.stButton>button:hover{background:#00d4ff18!important;border-color:#00d4ff!important;}
.stDownloadButton>button{background:#0d0d1a!important;color:#a78bfa!important;
  border:1px solid #2a2a45!important;border-radius:5px!important;
  font-family:'Share Tech Mono',monospace!important;font-size:.68rem!important;}
.stDownloadButton>button:hover{background:#a78bfa18!important;border-color:#a78bfa!important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:#080810;}
::-webkit-scrollbar-thumb{background:#1e1e35;border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PHYSICS CONSTANTS
# ══════════════════════════════════════════════════════════════════
MU_0=4*np.pi*1e-7; WIRE_RAD=1e-3; N_WP=420; N_GRID=17; N_3D=6

# ══════════════════════════════════════════════════════════════════
#  GEOMETRY
# ══════════════════════════════════════════════════════════════════
def wire_straight(L):
    t=np.linspace(-L/2,L/2,N_WP)
    return np.column_stack([np.zeros_like(t),np.zeros_like(t),t])

def wire_loop(R=0.4):
    th=np.linspace(0,2*np.pi,N_WP)
    return np.column_stack([R*np.cos(th),R*np.sin(th),np.zeros_like(th)])

def wire_solenoid(L,N_turns,R=0.3):
    th=np.linspace(0,2*np.pi*N_turns,4000); z=np.linspace(-L/2,L/2,4000)
    xyz=np.column_stack([R*np.cos(th),R*np.sin(th),z])
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))])
    s=np.linspace(0,arc[-1],N_WP)
    return np.column_stack([np.interp(s,arc,xyz[:,i]) for i in range(3)])

def apply_transform(wire,pos,rot_deg):
    rx,ry,rz=[np.radians(a) for a in rot_deg]
    Rx=np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry=np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz=np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return wire@(Rz@Ry@Rx).T+np.array(pos)

def build_wire(wd):
    m=wd["mode"]
    w=wire_straight(wd["L"]) if m=="Straight" else wire_loop() if m=="Loop" else wire_solenoid(wd["L"],wd["N"])
    return apply_transform(w,wd.get("pos",[0,0,0]),wd.get("rot",[0,0,0]))

# ══════════════════════════════════════════════════════════════════
#  BIOT-SAVART
# ══════════════════════════════════════════════════════════════════
def biot_savart(wire,I,obs):
    dl=np.diff(wire,axis=0).astype(np.float64)
    mid=(wire[:-1]+dl*.5).astype(np.float64)
    pre=MU_0*I/(4*np.pi)
    wr2=float(np.clip(np.linalg.norm(dl,axis=1).mean()*.3,WIRE_RAD,5*WIRE_RAD)**2)
    shp=obs.shape[:-1]; of=obs.reshape(-1,3).astype(np.float64)
    r=of[:,np.newaxis,:]-mid; rm=np.linalg.norm(r,axis=-1)
    dB=pre*np.cross(dl,r)/((rm**2+wr2)**1.5)[...,np.newaxis]
    return dB.sum(axis=1).reshape(shp+(3,))

def sumB(wires_I,obs):
    return sum(biot_savart(w,I,obs) for w,I in wires_I)

# ══════════════════════════════════════════════════════════════════
#  RK4 FIELD LINE
# ══════════════════════════════════════════════════════════════════
def trace_fl(wires_I,seed,ds=0.02,ns=200):
    def B(p): return sumB(wires_I,p.reshape(1,3))[0]
    pts=[seed.copy()]; p=seed.copy()
    for _ in range(ns):
        b1=B(p); n1=np.linalg.norm(b1)
        if n1<1e-20: break
        k1=b1/n1; b2=B(p+.5*ds*k1); n2=np.linalg.norm(b2)
        if n2<1e-20: break
        k2=b2/n2; b3=B(p+.5*ds*k2); n3=np.linalg.norm(b3)
        if n3<1e-20: break
        k3=b3/n3; b4=B(p+ds*k3); n4=np.linalg.norm(b4)
        if n4<1e-20: break
        k4=b4/n4; p=p+(ds/6)*(k1+2*k2+2*k3+k4)
        if np.linalg.norm(p)>1.5: break
        pts.append(p.copy())
    return np.array(pts)

# ══════════════════════════════════════════════════════════════════
#  ANALYTICAL
# ══════════════════════════════════════════════════════════════════
def analytics(T,mode,I,L,N,freq,ac):
    v={}
    if mode=="Straight":
        r=0.05
        v[T["b_r5"]]=(MU_0*I/(2*np.pi*r)*1e6,"µT")
        v[T["force"]]=(MU_0*I**2/(2*np.pi*r)*1e6,"µN/m")
    elif mode=="Loop":
        R=0.4;a=1e-3;Li=MU_0*R*(np.log(8*R/a)-2)
        v[T["b_ctr"]]=(MU_0*I/(2*R)*1e6,"µT")
        v[T["induc"]]=(Li*1e6,"µH")
        v[T["energy"]]=(0.5*Li*I**2*1e9,"nJ")
        if ac and freq>0: v[T["emf"]]=(Li*I*2*np.pi*freq*1e3,"mV")
    else:
        n=N/L;R=0.3;Li=MU_0*n**2*np.pi*R**2*L
        v[T["b_ctr"]]=(MU_0*n*I*1e3,"mT")
        v[T["induc"]]=(Li*1e3,"mH")
        v[T["energy"]]=(0.5*Li*I**2*1e6,"µJ")
        v[T["n_dens"]]=(n,"t/m")
        if ac and freq>0: v[T["emf"]]=(Li*I*2*np.pi*freq*1e3,"mV")
    return v

# ══════════════════════════════════════════════════════════════════
#  FIELD LINE SEEDS
# ══════════════════════════════════════════════════════════════════
def fl_seeds(plane,mode):
    if mode=="Loop":
        if plane=="xy":
            return [(0.42*np.cos(t),0.42*np.sin(t)) for t in np.linspace(0,2*np.pi,8,endpoint=False)]
        return [(0.42,z) for z in np.linspace(-0.6,0.6,5)]+[(-0.42,z) for z in np.linspace(-0.6,0.6,5)]+[(0.02,z) for z in np.linspace(-0.5,0.5,4)]
    if mode=="Solenoid":
        if plane=="xy": return []
        i=[(x,z) for x in [-0.15,0,0.15] for z in np.linspace(-0.42,0.42,4)]
        e=[(0.38,z) for z in np.linspace(-0.55,0.55,5)]+[(-0.38,z) for z in np.linspace(-0.55,0.55,5)]
        return i+e
    if plane=="xy": return [(r,0.0) for r in [0.08,0.18,0.32,0.50]]
    return []

# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB 2D
# ══════════════════════════════════════════════════════════════════
BG="#080810"; AC="#00d4ff"; AC2="#ff6b35"; AC3="#a78bfa"

def _wire_pts(w, plane):
    """Extract wire cross-section points for a given 2D plane."""
    if   plane=="xy": pts=w[np.abs(w[:,2])<.05]; return pts[:,0],pts[:,1]
    elif plane=="xz": pts=w[np.abs(w[:,1])<.05]; return pts[:,0],pts[:,2]
    else:             pts=w[np.abs(w[:,0])<.05]; return pts[:,1],pts[:,2]

def fig2d(U, V, Bu, Bv, wires_I, plane, mode, show_fl, cmap, title):
    """2D field — v2 quality: log-scaled arrows + patheffects glow on field lines."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor=BG)
    ax.set_facecolor(BG)

    # filled contour — field magnitude
    B_mag = np.sqrt(Bu**2 + Bv**2) + 1e-30
    cf = ax.contourf(U, V, B_mag, levels=20, cmap=cmap, alpha=0.88)

    # direction arrows — log-scaled so weak regions are still visible
    log_s = np.log1p(B_mag / B_mag.max())
    ax.quiver(U, V, Bu/B_mag*log_s, Bv/B_mag*log_s,
              color="white", scale=4.0, width=0.004, alpha=0.55,
              headwidth=4, headlength=5)

    # RK4 field lines with cyan glow
    if show_fl:
        for sx, sy in fl_seeds(plane, mode):
            s3 = (np.array([sx,sy,0.0]) if plane=="xy" else
                  np.array([sx,0.0,sy]) if plane=="xz" else
                  np.array([0.0,sx,sy]))
            fl = trace_fl(wires_I, s3)
            if len(fl) < 2: continue
            if   plane=="xy": px, py = fl[:,0], fl[:,1]
            elif plane=="xz": px, py = fl[:,0], fl[:,2]
            else:             px, py = fl[:,1], fl[:,2]
            mask = (np.abs(px)<0.72) & (np.abs(py)<0.72)
            if mask.sum() > 1:
                ax.plot(px[mask], py[mask], color=AC, lw=1.0, alpha=0.75,
                        path_effects=[
                            pe.Stroke(linewidth=2.5, foreground="#00d4ff18"),
                            pe.Normal()])

    # wire cross-section dots
    for i, (w, _) in enumerate(wires_I):
        xs, ys = _wire_pts(w, plane)
        if len(xs):
            ax.scatter(xs, ys, c=AC2 if i==0 else AC3,
                       s=22, zorder=6, linewidths=0.8, edgecolors="#ffffff33")

    # axes
    plane_axes = {"xy":("X [m]","Y [m]"), "xz":("X [m]","Z [m]"), "yz":("Y [m]","Z [m]")}
    xl, yl = plane_axes[plane]
    ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel(xl, color="#3a3a60", fontsize=7, labelpad=2)
    ax.set_ylabel(yl, color="#3a3a60", fontsize=7, labelpad=2)
    ax.set_title(title, color=AC, fontsize=9,
                 fontfamily="monospace", pad=5, fontweight="bold")
    ax.tick_params(colors="#3a3a60", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1a30"); sp.set_linewidth(0.5)

    # colorbar
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.03)
    cbar.ax.tick_params(colors="#5a5a90", labelsize=6)
    cbar.outline.set_edgecolor("#1a1a30")
    cbar.set_label("|B| [T]", color="#5a5a90", fontsize=6)

    fig.tight_layout(pad=0.4)
    return fig

# ══════════════════════════════════════════════════════════════════
#  PLOTLY 3D (interactive drag/rotate + hover)
# ══════════════════════════════════════════════════════════════════
def fig3d(wires_I, mode, show_fl):
    """3D field — Plotly interactive (drag/rotate/hover) with clearly visible cones."""
    l3 = np.linspace(-0.9, 0.9, N_3D)
    X, Y, Z = np.meshgrid(l3, l3, l3)
    B3d = sumB(wires_I, np.stack([X,Y,Z],-1))
    Bx, By, Bz = B3d[...,0], B3d[...,1], B3d[...,2]
    Bm = np.sqrt(Bx**2 + By**2 + Bz**2) + 1e-30

    traces = [go.Cone(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        u=(Bx/Bm).ravel(), v=(By/Bm).ravel(), w=(Bz/Bm).ravel(),
        # sizemode="scaled" + sizeref<1 → 大錐體，與格點間距成比例
        colorscale=[
            [0.0, "#0066cc"], [0.3, "#00aaff"],
            [0.6, "#00ffee"], [1.0, "#ffffff"],
        ],
        cmin=0, cmax=1,
        sizemode="scaled",
        sizeref=0.55,          # 0~1，越小越大；0.55 約佔格距 45%
        showscale=False,
        opacity=1.0,
        anchor="tail",         # 尾部固定在格點，方向朝場方向
        hovertemplate=(
            "x=%{x:.2f}  y=%{y:.2f}  z=%{z:.2f}<br>"
            "B=(%{u:.3f}, %{v:.3f}, %{w:.3f})<extra></extra>"
        ),
    )]

    # 導線路徑
    for i, (w, _) in enumerate(wires_I):
        traces.append(go.Scatter3d(
            x=w[:,0], y=w[:,1], z=w[:,2],
            mode="lines",
            line=dict(color=AC2 if i==0 else AC3, width=5),
            name=f"Wire {i+1}",
            hovertemplate=f"Wire {i+1}<extra></extra>",
        ))

    # 磁力線
    if show_fl:
        seeds = []
        if mode == "Loop":
            seeds = [np.array([0.42*np.cos(t), 0.42*np.sin(t), 0.0])
                     for t in np.linspace(0, 2*np.pi, 6, endpoint=False)]
        elif mode == "Solenoid":
            for ang in np.linspace(0, 2*np.pi, 4, endpoint=False):
                seeds.append(np.array([0.32*np.cos(ang), 0.32*np.sin(ang), 0.0]))
            seeds += [np.array([0.01, 0.0, z]) for z in np.linspace(-0.28, 0.28, 3)]
        else:
            seeds = [np.array([r, 0.0, 0.0]) for r in [0.07, 0.15, 0.28, 0.44]]
        for seed in seeds:
            fl = trace_fl(wires_I, seed, ds=0.025, ns=130)
            if len(fl) > 2:
                traces.append(go.Scatter3d(
                    x=fl[:,0], y=fl[:,1], z=fl[:,2],
                    mode="lines",
                    line=dict(color=AC, width=2),
                    showlegend=False,
                    hovertemplate="Field line<extra></extra>",
                ))

    layout = go.Layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        scene=dict(
            bgcolor=BG,
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, title=""),
            # 初始視角接近原版 matplotlib 的 elev=20, azim=-60
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=440,
        showlegend=False,
    )
    return go.Figure(data=traces, layout=layout)

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
if "wl" not in st.session_state:
    st.session_state.wl=[{"mode":"Solenoid","I":40,"L":1.0,"N":12,"pos":[0,0,0],"rot":[0,0,0]}]
if "phase" not in st.session_state: st.session_state.phase=0.0
if "lang"  not in st.session_state: st.session_state.lang="繁中"

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Language picker — flags in label
    lang_opts=list(LANGS.keys())
    lang_display=[f"{LANGS[k]['flag']} {k}" for k in lang_opts]
    cur_idx=lang_opts.index(st.session_state.lang)
    lang_sel=st.selectbox("🌐",lang_display,index=cur_idx,label_visibility="collapsed")
    new_lang=lang_opts[lang_display.index(lang_sel)]
    if new_lang!=st.session_state.lang:
        st.session_state.lang=new_lang; st.rerun()
    T=LANGS[st.session_state.lang]

    st.markdown(f'<div class="app-title">⚡ {T["title"]}</div>'
                f'<div class="app-sub">{T["subtitle"]}</div>'
                f'<div class="hdiv"></div>',unsafe_allow_html=True)

    # Geometry
    st.markdown(f'<div class="slbl">{T["geometry"]}</div>',unsafe_allow_html=True)
    mk=[T["straight"],T["loop"],T["solenoid"]]; mv=["Straight","Loop","Solenoid"]
    ms=st.radio(" ",mk,index=2,horizontal=True,label_visibility="collapsed")
    mode=mv[mk.index(ms)]

    # Parameters
    st.markdown(f'<div class="slbl">{T["parameters"]}</div>',unsafe_allow_html=True)
    I_val=st.slider(T["current"],1.0,100.0,40.0,1.0)
    L_val=st.slider(T["length"], 0.2,2.0,  1.0, .05)
    N_val=st.slider(T["turns"],  1,  40,   12,  1)

    # AC
    st.markdown(f'<div class="slbl">{T["ac_sec"]}</div>',unsafe_allow_html=True)
    ac_on=st.checkbox(T["ac_on"],value=False)
    freq=st.slider(T["freq"],0,120,50,5,disabled=not ac_on)
    if ac_on: st.caption(T["ac_note"])

    # Display
    st.markdown(f'<div class="slbl">{T["display"]}</div>',unsafe_allow_html=True)
    dc1,dc2=st.columns(2)
    show_fl=dc1.checkbox(T["show_fl"],value=True)
    cmap=dc2.selectbox(" ",["magma","plasma","viridis","inferno","hot"],label_visibility="collapsed")

    # Wire manager
    st.markdown(f'<div class="slbl">{T["wires"]}</div>',unsafe_allow_html=True)
    for i,wd in enumerate(st.session_state.wl):
        ca,cb=st.columns([5,1])
        clr="#00d4ff" if i==0 else "#a78bfa"
        ca.markdown(f'<span class="wpill" style="color:{clr}">#{i+1} {wd["mode"][:3].upper()} I={wd["I"]:.0f}A</span>',
                    unsafe_allow_html=True)
        if cb.button(T["w_del"],key=f"rm{i}") and i>0:
            st.session_state.wl.pop(i); st.rerun()

    with st.expander(T["add_wire"]):
        nm=st.selectbox(T["w_type"],["Straight","Loop","Solenoid"],key="nm")
        nI=st.number_input(T["w_I"],1.0,200.0,20.0,key="nI")
        nL=st.number_input(T["w_L"],0.1,3.0,1.0,key="nL")
        nN=st.number_input(T["w_N"],1,60,8,key="nN",step=1)
        st.caption(f"📍 {T['w_pos']}")
        p1,p2,p3=st.columns(3)
        px_=p1.number_input("X",-1.5,1.5,0.0,.1,key="px_")
        py_=p2.number_input("Y",-1.5,1.5,0.0,.1,key="py_")
        pz_=p3.number_input("Z",-1.5,1.5,0.0,.1,key="pz_")
        st.caption(f"🔄 {T['w_rot']}")
        r1,r2,r3=st.columns(3)
        rx_=r1.number_input("Rx",-180.,180.,0.,5.,key="rx_")
        ry_=r2.number_input("Ry",-180.,180.,0.,5.,key="ry_")
        rz_=r3.number_input("Rz",-180.,180.,90.,5.,key="rz_")
        if st.button(T["w_add"],use_container_width=True):
            st.session_state.wl.append({"mode":str(nm),"I":float(nI),"L":float(nL),"N":int(nN),
                "pos":[px_,py_,pz_],"rot":[rx_,ry_,rz_]}); st.rerun()

    if len(st.session_state.wl)>1:
        if st.button(T["clear_extra"],use_container_width=True):
            st.session_state.wl=[st.session_state.wl[0]]; st.rerun()

    # Scene I/O
    st.markdown(f'<div class="slbl">{T["io_sec"]}</div>',unsafe_allow_html=True)
    scene={"wires":st.session_state.wl,"mode":mode,"I":I_val,"L":L_val,"N":N_val,
           "freq":freq,"ac":ac_on,"fl":show_fl,"cmap":cmap,"lang":st.session_state.lang}
    st.download_button(T["save_json"],data=json.dumps(scene,indent=2,ensure_ascii=False).encode(),
                       file_name="em_scene.json",mime="application/json",use_container_width=True)
    up=st.file_uploader(T["load_json"],type=["json"],label_visibility="collapsed")
    if up:
        sc=json.load(up)
        wl=sc.get("wires",st.session_state.wl)
        for wd in wl: wd.setdefault("pos",[0,0,0]); wd.setdefault("rot",[0,0,0])
        st.session_state.wl=wl
        if sc.get("lang") in LANGS: st.session_state.lang=sc["lang"]
        st.rerun()

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
T=LANGS[st.session_state.lang]

st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:14px;margin-bottom:4px">
  <span style="font-family:'Share Tech Mono',monospace;font-size:1.4rem;color:#00d4ff;
               letter-spacing:3px;text-shadow:0 0 14px rgba(0,212,255,.35)">
    ⚡ {T['title'].upper()}
  </span>
  <span style="font-family:'Share Tech Mono',monospace;font-size:.58rem;color:#2a2a50;letter-spacing:2px">
    {T['subtitle'].upper()}
  </span>
</div>
<div style="height:1px;background:linear-gradient(90deg,#00d4ff55,transparent);margin-bottom:10px"></div>
""",unsafe_allow_html=True)

# Sync primary wire
st.session_state.wl[0].update({"mode":mode,"I":I_val,"L":L_val,"N":N_val})

# AC phase
if ac_on and freq>0: st.session_state.phase+=0.06
phase=st.session_state.phase

# Build wires_I
wires_I=[]
for wd in st.session_state.wl:
    I=wd["I"]*(math.cos(2*math.pi*freq*phase) if (ac_on and freq>0) else 1.0)
    wires_I.append((build_wire(wd),I))

# AC refresh banner
if ac_on and freq>0:
    st.markdown(f"""<meta http-equiv="refresh" content="1">
    <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;
                color:#ff6b3599;margin-bottom:6px;letter-spacing:2px">
    {T['ac_live']} — AUTO REFRESH · f={freq} Hz · φ={phase:.2f} rad</div>""",
    unsafe_allow_html=True)

# Compute fields
with st.spinner("⚡"):
    l=np.linspace(-0.7,0.7,N_GRID); U,V=np.meshgrid(l,l); z=np.zeros_like(U)
    Bxy=sumB(wires_I,np.stack([U,V,z],-1))
    Bxz=sumB(wires_I,np.stack([U,z,V],-1))
    Byz=sumB(wires_I,np.stack([z,U,V],-1))

# ── 2×2 plots
c1,c2=st.columns(2,gap="small")
with c1:
    st.markdown('<div class="pw">',unsafe_allow_html=True)
    st.plotly_chart(fig3d(wires_I,mode,show_fl), use_container_width=True,
                    config={"displaylogo":False,
                            "modeBarButtonsToRemove":["toImage","resetCameraLastSave3d"]})
    st.markdown('</div><div class="pw" style="margin-top:7px">',unsafe_allow_html=True)
    fxy=fig2d(U,V,Bxy[...,0],Bxy[...,1],wires_I,"xy",mode,show_fl,cmap,T["plane_xy"])
    st.pyplot(fxy,use_container_width=True); plt.close(fxy)
    st.markdown('</div>',unsafe_allow_html=True)
with c2:
    st.markdown('<div class="pw">',unsafe_allow_html=True)
    fxz=fig2d(U,V,Bxz[...,0],Bxz[...,2],wires_I,"xz",mode,show_fl,cmap,T["plane_xz"])
    st.pyplot(fxz,use_container_width=True); plt.close(fxz)
    st.markdown('</div><div class="pw" style="margin-top:7px">',unsafe_allow_html=True)
    fyz=fig2d(U,V,Byz[...,1],Byz[...,2],wires_I,"yz",mode,show_fl,cmap,T["plane_yz"])
    st.pyplot(fyz,use_container_width=True); plt.close(fyz)
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown("""<div class="lr" style="margin-top:4px">
  <div class="li"><div class="ld" style="background:#ff6b35"></div>Wire #1</div>
  <div class="li"><div class="ld" style="background:#a78bfa"></div>Wire #2+</div>
  <div class="li"><div class="ld" style="background:#00d4ff"></div>Field lines (RK4)</div>
  <div class="li"><div class="ld" style="background:#fff8"></div>Direction arrows</div>
</div>""",unsafe_allow_html=True)

# ── Analytics
st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,#1e1e35,transparent);margin:12px 0 7px"></div>'
            f'<div class="slbl">{T["analytics"]}</div>',unsafe_allow_html=True)
nums=analytics(T,mode,I_val,L_val,N_val,freq,ac_on)
pal=["b","b","p","p","b","p"]*3
html='<div class="mr">'
for (k,(v,u)),c in zip(nums.items(),pal):
    html+=f'<div class="mc {c}"><div class="ml">{k}</div><div class="mv">{v:.4g}<span style="font-size:.65rem;color:#5a5a90"> {u}</span></div></div>'
st.markdown(html+'</div>',unsafe_allow_html=True)

# ── B-field query
st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,transparent,#1e1e35,transparent);margin:12px 0 7px"></div>'
            f'<div class="slbl">{T["query"]}</div>',unsafe_allow_html=True)
qc1,qc2,qc3,qc4=st.columns([1,1,1,3])
qx=qc1.number_input("X (m)",-0.7,0.7,0.0,.05,key="qx",format="%.2f")
qy=qc2.number_input("Y (m)",-0.7,0.7,0.0,.05,key="qy",format="%.2f")
qz=qc3.number_input("Z (m)",-0.7,0.7,0.0,.05,key="qz",format="%.2f")
Bq=sumB(wires_I,np.array([[qx,qy,qz]]))[0]*1e6; Bqm=np.linalg.norm(Bq)
qc4.markdown(f"""<div class="bqbox">
  <div class="bqtitle">{T['bq_title']}  ({qx:.2f},{qy:.2f},{qz:.2f}) m</div>
  <div class="bvals">
    <div class="bvi"><span class="bvl">Bx [µT]</span><span class="bvn">{Bq[0]:+.4f}</span></div>
    <div class="bvi"><span class="bvl">By [µT]</span><span class="bvn">{Bq[1]:+.4f}</span></div>
    <div class="bvi"><span class="bvl">Bz [µT]</span><span class="bvn">{Bq[2]:+.4f}</span></div>
    <div class="bvi"><span class="bvl">|B| [µT]</span><span class="bvn tot">{Bqm:.4f}</span></div>
  </div></div>""",unsafe_allow_html=True)

# ── Export row
st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#1e1e35,transparent);margin:12px 0 7px"></div>',
            unsafe_allow_html=True)
ex1,ex2=st.columns(2)

# CSV
with ex1:
    rows=[["plane","x","y","Bx_uT","By_uT","Bz_uT","Bmag_uT"]]
    for pn,Bd in [("XY",Bxy),("XZ",Bxz),("YZ",Byz)]:
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                bv=Bd[i,j]*1e6
                rows.append([pn,f"{U[i,j]:.4f}",f"{V[i,j]:.4f}",
                              f"{bv[0]:.6f}",f"{bv[1]:.6f}",f"{bv[2]:.6f}",f"{np.linalg.norm(bv):.6f}"])
    buf=io.StringIO(); csv.writer(buf).writerows(rows)
    st.download_button(T["export_csv"],data=buf.getvalue().encode(),
                       file_name=f"em_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv",use_container_width=True)

# Screenshot (XY PNG)
with ex2:
    fss=fig2d(U,V,Bxy[...,0],Bxy[...,1],wires_I,"xy",mode,show_fl,cmap,T["plane_xy"])
    buf2=io.BytesIO(); fss.savefig(buf2,format="png",dpi=130,bbox_inches="tight",facecolor=BG)
    plt.close(fss); buf2.seek(0)
    st.download_button(T["screenshot"],data=buf2,
                       file_name=f"em_snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                       mime="image/png",use_container_width=True)

# Footer
st.markdown(f"""<div style="margin-top:24px;padding-top:10px;border-top:1px solid #1a1a30;
  text-align:center;font-family:'Share Tech Mono',monospace;
  font-size:.58rem;color:#2a2a50;letter-spacing:2px">{T['footer']}</div>""",
  unsafe_allow_html=True)

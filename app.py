"""
GlucoCheck Pro+ — Production-grade Streamlit app
Fixes: unified scroll (no sidebar), OpenAI chat, statsmodels crash removed,
       stylable_container removed, full dark theme throughout.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import datetime
import time
import sys
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_calendar import calendar

# Optional OpenAI
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ── Path / auth setup ──────────────────────────────────────────────────────────
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, "auth"))

try:
    from auth_utils import (initialize_user_db, register_user, verify_user,
                             generate_token)
except ImportError:
    def initialize_user_db(): pass
    def register_user(u, p): return False
    def verify_user(u, p):   return False
    def generate_token(u):   return "tok"

# clean leftover artefacts
for _f in ["confusion_matrix.html"]:
    _p = os.path.join(current_dir, _f)
    if os.path.exists(_p):
        try: os.remove(_p)
        except: pass

# ── Page config ────────────────────────────────────────────────────────────────
# Collapse the native sidebar entirely — layout is done with st.columns instead
st.set_page_config(
    page_title="GlucoCheck Pro+",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CSS  — dark theme + hide native sidebar chrome
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset / base ── */
:root {
    --bg:       #0d1117;
    --panel:    #161b22;
    --card:     #1c2128;
    --border:   rgba(108,99,255,.22);
    --purple:   #6C63FF;
    --dpurple:  #4D44DB;
    --text:     rgba(255,255,255,.88);
    --muted:    rgba(255,255,255,.48);
    --success:  #3fb950;
    --warning:  #d29922;
    --danger:   #f85149;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide native Streamlit sidebar & hamburger button completely ── */
[data-testid="stSidebar"]          { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
button[kind="header"]              { display: none !important; }

/* ── Remove top padding / footer ── */
.block-container { padding: 1rem 1.4rem 2rem !important; max-width: 100% !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Make entire page one scroll region ── */
html, body { overflow-y: auto !important; height: auto !important; }
[data-testid="stAppViewContainer"],
section[data-testid="stMain"] { overflow: visible !important; height: auto !important; }

/* ── Global dark text overrides ── */
p, span, label, li, h1, h2, h3, h4,
.stMarkdown, .stText { color: var(--text) !important; }
hr { border-color: var(--border) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,var(--purple),var(--dpurple)) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 10px 22px !important;
    font-weight: 600 !important; font-size: .9rem !important;
    box-shadow: 0 4px 14px rgba(108,99,255,.35) !important;
    transition: all .25s ease !important; cursor: pointer !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 7px 22px rgba(108,99,255,.5) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"]                       { color: var(--muted) !important; font-weight: 500 !important; padding: 8px 16px !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--purple) !important; border-bottom: 2px solid var(--purple) !important; }
[data-testid="stTabs"] [data-testid="stTabsContainer"]    { border-bottom: 1px solid var(--border) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 500 !important; }

/* ── Forms ── */
[data-testid="stForm"] { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 20px !important; }
[data-testid="stForm"] input, [data-testid="stForm"] textarea,
[data-testid="stForm"] select { background: rgba(255,255,255,.05) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }

/* ── Text inputs (outside forms) ── */
[data-testid="stTextInput"] input { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text) !important; padding: 10px 14px !important; }
[data-testid="stTextInput"] input:focus { border-color: var(--purple) !important; box-shadow: 0 0 0 3px rgba(108,99,255,.18) !important; }
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }

/* ── Selectbox / multiselect ── */
[data-baseweb="select"] > div { background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: var(--purple) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { background: rgba(108,99,255,.1) !important; border: 1px solid rgba(108,99,255,.28) !important; border-radius: 10px !important; }
[data-testid="stAlert"] p { color: var(--text) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button { background: linear-gradient(135deg,#238636,#2ea043) !important; border-radius: 10px !important; }

/* ═══════════════════════════════════════════
   LEFT PANEL  (replaces native sidebar)
═══════════════════════════════════════════ */
.left-panel {
    background: var(--panel);
    border-right: 1px solid var(--border);
    min-height: 100vh;
    padding: 20px 16px;
    position: sticky;
    top: 0;
}

/* Brand / user card */
.brand-card {
    background: rgba(108,99,255,.12); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 14px 14px; text-align: center; margin-bottom: 18px;
}
.brand-card .b-icon   { font-size: 2.2rem; display: block; margin-bottom: 5px; }
.brand-card .b-title  { font-size: 1.1rem; font-weight: 700; color: #fff; display: block; margin-bottom: 2px; }
.brand-card .b-sub    { font-size: .7rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }

.user-card {
    background: rgba(108,99,255,.1); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px; text-align: center; margin-bottom: 14px;
}
.user-card .u-icon { font-size: 2rem; display: block; margin-bottom: 4px; }
.user-card .u-name { font-size: .98rem; font-weight: 600; color: #fff; }
.user-card .u-sub  { font-size: .72rem; color: var(--muted); }

/* Auth tabs */
.auth-tabs  { display: flex; gap: 4px; background: rgba(255,255,255,.05); border-radius: 12px; padding: 4px; margin-bottom: 18px; }
.auth-tab   { flex: 1; text-align: center; padding: 8px 0; border-radius: 9px; font-size: .84rem; font-weight: 600; color: var(--muted); }
.auth-tab.on { background: linear-gradient(135deg,var(--purple),var(--dpurple)); color: #fff; box-shadow: 0 3px 10px rgba(108,99,255,.4); }

.field-lbl  { font-size: .75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1.1px; margin-bottom: 4px; display: block; }
.privacy    { text-align: center; font-size: .72rem; color: rgba(255,255,255,.25); padding: 8px 0; }

/* ═══════════════════════════════════════════
   MAIN CONTENT
═══════════════════════════════════════════ */

/* App header */
.app-hdr {
    background: linear-gradient(135deg,#0d1117,#161b22,#1c2128);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 24px 28px; margin-bottom: 22px;
    display: flex; align-items: center; gap: 16px;
}
.app-hdr-icon { font-size: 2.6rem; line-height: 1; }
.app-hdr-text h1 { font-size: 1.7rem; font-weight: 700; color: #fff; margin: 0 0 3px; }
.app-hdr-text p  { font-size: .85rem; color: var(--muted); margin: 0; }

/* Risk cards */
.risk-card {
    background: var(--card); border-radius: 12px; padding: 18px;
    border-left: 5px solid var(--purple); margin-bottom: 12px;
}
.risk-card .rc-title { font-size: .8rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; margin: 0 0 6px; }
.risk-card .rc-val   { font-size: 1.6rem; font-weight: 700; color: #fff; margin: 0; }
.risk-card .rc-icon  { float: right; font-size: 1.4rem; }

/* Section header */
.sec-hdr { text-align: center; margin: 30px 0 20px; }
.sec-hdr h2 { font-size: 1.55rem; font-weight: 700; color: #fff; margin-bottom: 5px; }
.sec-hdr p  { color: var(--muted); font-size: .9rem; margin: 0; }
.sec-bar { width: 44px; height: 4px; background: linear-gradient(90deg,var(--purple),var(--dpurple)); border-radius: 2px; margin: 7px auto 0; }

/* Feature / step cards */
.feat-card {
    background: var(--card); border-radius: 14px; padding: 22px 18px;
    border-top: 4px solid; border-left: 1px solid var(--border);
    border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
    text-align: center; height: 100%; transition: transform .22s, box-shadow .22s;
}
.feat-card:hover { transform: translateY(-4px); box-shadow: 0 14px 36px rgba(0,0,0,.45); }
.feat-card .fi { font-size: 2.1rem; margin-bottom: 10px; display: block; }
.feat-card h3  { font-size: .97rem; font-weight: 600; color: #fff; margin-bottom: 7px; }
.feat-card p   { font-size: .84rem; color: var(--muted); line-height: 1.5; margin: 0; }

.step-card { background: var(--card); border-radius: 13px; padding: 20px 15px; text-align: center; border: 1px solid rgba(108,99,255,.14); }
.step-num  { font-size: 1.8rem; font-weight: 700; color: var(--purple); display: block; margin-bottom: 5px; }
.step-card h4 { font-size: .93rem; font-weight: 600; color: #fff; margin-bottom: 5px; }
.step-card p  { font-size: .82rem; color: var(--muted); margin: 0; }

/* Hero */
.hero {
    background: linear-gradient(135deg,#0d1117,#161b22 50%,#1c2128);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 58px 44px 50px; margin-bottom: 24px;
    text-align: center; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40%; left: -20%; width: 160%; height: 160%;
    background: radial-gradient(circle at 60% 40%, rgba(108,99,255,.13) 0%, transparent 55%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block; background: rgba(108,99,255,.2);
    border: 1px solid rgba(108,99,255,.4); color: #c4c0ff;
    padding: 5px 18px; border-radius: 50px; font-size: .79rem;
    font-weight: 500; margin-bottom: 16px; letter-spacing: .8px; text-transform: uppercase;
}
.hero h1 { font-size: 2.9rem; font-weight: 700; color: #fff; margin: 0 0 12px; line-height: 1.15; }
.hero h1 span { background: linear-gradient(90deg,#a78bfa,#818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p  { font-size: 1.03rem; color: rgba(255,255,255,.63); max-width: 520px; margin: 0 auto 28px; font-weight: 300; }
.hero-pills { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.hero-pill  { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); color: rgba(255,255,255,.8); padding: 7px 18px; border-radius: 50px; font-size: .84rem; font-weight: 500; }

/* Stats bar */
.stats-bar {
    background: linear-gradient(135deg,var(--purple),var(--dpurple));
    border-radius: 14px; padding: 24px 16px; margin-bottom: 24px;
    display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 14px;
}
.stat .n { font-size: 1.85rem; font-weight: 700; color: #fff; display: block; }
.stat .l { font-size: .74rem; color: rgba(255,255,255,.65); letter-spacing: .5px; display: block; text-align: center; }

/* Chat */
.chat-box {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px; margin-bottom: 14px;
    max-height: 400px; overflow-y: auto;
}
.msg-user {
    background: linear-gradient(135deg,var(--purple),var(--dpurple)); color: #fff;
    border-radius: 16px 16px 2px 16px; padding: 10px 14px; margin: 8px 0 8px auto;
    max-width: 80%; word-wrap: break-word; display: block;
    box-shadow: 0 2px 8px rgba(108,99,255,.3);
}
.msg-ai {
    background: var(--panel); color: var(--text);
    border-radius: 16px 16px 16px 2px; padding: 10px 14px; margin: 8px auto 8px 0;
    max-width: 80%; word-wrap: break-word; display: block;
    border: 1px solid var(--border);
}
.msg-t { font-size: .66em; opacity: .5; margin-top: 3px; }

/* History / med cards */
.hist-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 11px; padding: 13px 16px; margin-bottom: 8px;
}
.med-card {
    background: var(--card); border-left: 4px solid var(--purple);
    border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
    border-radius: 11px; padding: 13px 16px; margin-bottom: 9px;
}
.med-card h4 { color: #fff; margin: 0 0 6px; font-size: .95rem; }
.med-card p  { color: var(--muted); margin: 1px 0; font-size: .83rem; }

@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
.fade-in { animation: fadeUp .4s ease-out; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_ss("username",       None)
_ss("token",          None)
_ss("chat_history",   [])
_ss("medications",    [])
_ss("health_history", [])
_ss("auth_mode",      "Login")


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model():
    mp = os.path.join(current_dir, "model.pkl")
    sp = os.path.join(current_dir, "scaler.pkl")
    m = s = None
    try:
        if os.path.exists(mp): m = joblib.load(mp)
        if os.path.exists(sp): s = joblib.load(sp)
    except Exception as e:
        st.error(f"Model load error: {e}")
    return m, s

ML_MODEL, ML_SCALER = load_model()


# =============================================================================
# OPENAI CHAT
# =============================================================================
class ChatAssistant:
    SYSTEM = (
        "You are a knowledgeable diabetes specialist AI. "
        "Give accurate, evidence-based answers about diabetes prevention, management, "
        "diet, exercise, and medication. Keep replies concise (≤150 words). "
        "Always remind users to consult a qualified healthcare professional."
    )

    def __init__(self):
        self.client = None
        self.error  = None
        if not _OPENAI_AVAILABLE:
            self.error = "openai package not installed (add to requirements.txt)."
            return
        key = None
        try:   key = st.secrets.get("OPENAI_API_KEY")
        except: pass
        if not key: key = os.getenv("OPENAI_API_KEY")
        if not key:
            self.error = "OPENAI_API_KEY not found in secrets or environment."
            return
        try:
            self.client = _OpenAI(api_key=key)
        except Exception as e:
            self.error = str(e)

    def ask(self, question: str, context: str = "") -> str:
        if not self.client:
            return f"⚠️ Chat unavailable: {self.error}"
        messages = [{"role": "system", "content": self.SYSTEM}]
        if context:
            messages.append({"role": "system", "content": f"User health context: {context}"})
        messages.append({"role": "user", "content": question})
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.65,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"


@st.cache_resource
def get_chat_assistant():
    return ChatAssistant()


# =============================================================================
# HELPERS
# =============================================================================
_RISK_COLORS = {"Very Low":"#3fb950","Low":"#8BC34A","Moderate":"#d29922",
                "High":"#FF9800","Very High":"#f85149"}

def _risk(prob, bmi, age):
    ag  = "Child/Teen" if age < 18 else ("Senior" if age > 60 else "Adult")
    bmc = ("Underweight" if bmi<18.5 else "Normal" if bmi<25
           else "Overweight" if bmi<30 else "Obese")
    tiers = [
        (.2, "Very Low","#3fb950","😊",
         ["Balanced whole-food diet.","Limit processed sugars.","Stay hydrated."],
         ["30-min brisk walk 5×/week.","Strength training 2-3×/week."],
         ["Multivitamin.","Omega-3 fatty acids."]),
        (.4,"Low","#8BC34A","🙂",
         ["Focus on portion control.","Increase fibre intake.","Cut sugary drinks."],
         ["150 min moderate exercise/week.","Cycling or swimming."],
         ["Vitamin D."]),
        (.6,"Moderate","#d29922","😐",
         ["Limit refined carbs.","Lean protein + healthy fats.","See a nutritionist."],
         ["200–250 min moderate/week.","Cardio + strength training."],
         ["Chromium picolinate.","Magnesium."]),
        (.8,"High","#FF9800","😟",
         ["Low-GI diet.","Eliminate sugary snacks.","Work with a dietitian."],
         ["250–300 min moderate/week.","Monitor blood sugar pre/post exercise."],
         ["Berberine (doctor approval).","Alpha-lipoic acid."]),
        (2.,"Very High","#f85149","❗",
         ["Medical nutrition therapy.","Zero processed food.","Regular BGL monitoring."],
         ["Daily activity even short walks.","Supervised exercise programme."],
         ["Supplements only under supervision.","CoQ10."]),
    ]
    for t,rl,col,ic,di,ex,su in tiers:
        if prob < t:
            return dict(risk=rl,color=col,icon=ic,diet=di,exercise=ex,
                        supplements=su,bmi_cat=bmc,age_grp=ag)


def _gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100,
        domain={"x":[0,1],"y":[0,1]},
        title={"text":"Risk Score (%)","font":{"size":15,"color":"white"}},
        number={"font":{"color":"white","size":28},"suffix":"%"},
        gauge={
            "axis":{"range":[None,100],"tickcolor":"#555"},
            "bar":{"color":color},
            "bgcolor":"#1c2128","borderwidth":1,"bordercolor":"#30363d",
            "steps":[
                {"range":[0,20],"color":"#0d2818"},{"range":[20,40],"color":"#182d18"},
                {"range":[40,60],"color":"#2e2400"},{"range":[60,80],"color":"#2d1500"},
                {"range":[80,100],"color":"#2d0000"},
            ],
            "threshold":{"line":{"color":"red","width":3},"thickness":.75,"value":prob*100},
        }
    ))
    fig.update_layout(height=260,margin=dict(l=10,r=10,t=40,b=10),
                      paper_bgcolor="#161b22",plot_bgcolor="#161b22",font_color="white")
    return fig


def _dark_fig(fig):
    fig.update_layout(paper_bgcolor="#161b22",plot_bgcolor="#161b22",
                      font_color="white",
                      xaxis=dict(gridcolor="#21262d",linecolor="#30363d"),
                      yaxis=dict(gridcolor="#21262d",linecolor="#30363d"))
    return fig


def _pdf(prob, rec, age, bmi, glucose, bp, skin, insulin, ped, preg):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold",20); c.setFillColorRGB(.42,.39,1.)
    c.drawString(50,h-50,"GlucoCheck Pro+ — Health Report")
    c.setFont("Helvetica",10); c.setFillColorRGB(.55,.55,.55)
    c.drawString(50,h-66,f"Generated: {datetime.date.today()}")
    y = h-105; c.setFont("Helvetica-Bold",12); c.setFillColorRGB(0,0,0)
    c.drawString(50,y,"Patient Metrics")
    c.setFont("Helvetica",10)
    for ln in [f"Age: {age} yrs | BMI: {bmi:.1f} ({rec['bmi_cat']}) | Glucose: {glucose} mg/dL",
               f"Blood Pressure: {bp} mmHg | Skin Thickness: {skin} mm | Insulin: {insulin} μU/mL",
               f"Diabetes Pedigree: {ped:.3f} | Pregnancies: {preg}"]:
        y -= 15; c.drawString(60,y,ln)
    y -= 25; c.setFont("Helvetica-Bold",12); c.drawString(50,y,"Risk Assessment")
    c.setFont("Helvetica",10); y-=15
    c.drawString(60,y,f"Probability: {prob*100:.1f}%   Risk Level: {rec['risk']}")
    for section, items in [("Diet",rec["diet"]),("Exercise",rec["exercise"]),("Supplements",rec["supplements"])]:
        y -= 22; c.setFont("Helvetica-Bold",11); c.drawString(50,y,f"{section}:")
        c.setFont("Helvetica",10)
        for item in items: y-=13; c.drawString(62,y,f"• {item}")
    c.setFont("Helvetica-Oblique",8); c.setFillColorRGB(.5,.5,.5)
    c.drawString(50,42,"Disclaimer: This is a risk assessment tool only. Always consult a qualified healthcare professional.")
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()


def _time_for_period(period, end=False):
    t = {"Morning":("08:00:00","09:00:00"),"Afternoon":("12:00:00","13:00:00"),
         "Evening":("18:00:00","19:00:00"),"Night":("21:00:00","22:00:00")}
    return t[period][1 if end else 0]


# =============================================================================
# LEFT PANEL  (used for both auth and logged-in sidebar)
# =============================================================================
def _panel_auth():
    st.markdown("""
    <div class="brand-card">
        <span class="b-icon">🏥</span>
        <span class="b-title">GlucoCheck Pro+</span>
        <span class="b-sub">Diabetes Management</span>
    </div>""", unsafe_allow_html=True)

    mode = st.session_state.auth_mode
    on_l = "on" if mode=="Login"    else ""
    on_r = "on" if mode=="Register" else ""
    st.markdown(f"""
    <div class="auth-tabs">
        <div class="auth-tab {on_l}">🔑 Login</div>
        <div class="auth-tab {on_r}">✨ Register</div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("Login",    use_container_width=True, key="sw_login"):
            st.session_state.auth_mode = "Login";    st.rerun()
    with col_r:
        if st.button("Register", use_container_width=True, key="sw_reg"):
            st.session_state.auth_mode = "Register"; st.rerun()

    st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

    if mode == "Login":
        st.markdown('<span class="field-lbl">Username</span>', unsafe_allow_html=True)
        u = st.text_input("u", placeholder="Your username", label_visibility="collapsed", key="l_u")
        st.markdown('<span class="field-lbl">Password</span>', unsafe_allow_html=True)
        p = st.text_input("p", placeholder="Your password", type="password", label_visibility="collapsed", key="l_p")
        if st.button("🔑 Sign In", use_container_width=True, key="do_login"):
            if verify_user(u, p):
                st.session_state.username = u
                st.session_state.token    = generate_token(u)
                st.rerun()
            else:
                st.error("Invalid username or password.")
    else:
        st.markdown('<span class="field-lbl">Username</span>', unsafe_allow_html=True)
        nu  = st.text_input("nu", placeholder="Choose a username",  label_visibility="collapsed", key="r_u")
        st.markdown('<span class="field-lbl">Password</span>', unsafe_allow_html=True)
        np_ = st.text_input("np", placeholder="Create a password",  type="password", label_visibility="collapsed", key="r_p")
        st.markdown('<span class="field-lbl">Confirm Password</span>', unsafe_allow_html=True)
        cp  = st.text_input("cp", placeholder="Repeat password",    type="password", label_visibility="collapsed", key="r_cp")
        if st.button("✨ Create Account", use_container_width=True, key="do_reg"):
            if not nu or not np_:
                st.error("Fill in all fields.")
            elif np_ != cp:
                st.error("Passwords don't match.")
            elif register_user(nu, np_):
                st.success("Account created! Please sign in.")
                st.session_state.auth_mode = "Login"; st.rerun()
            else:
                st.error("Username already taken.")

    st.markdown('<div class="privacy">🔒 Your data is private & secure</div>', unsafe_allow_html=True)


def _panel_app(username):
    st.markdown(f"""
    <div class="user-card">
        <span class="u-icon">👤</span>
        <span class="u-name">{username}</span>
        <span class="u-sub">Active session</span>
    </div>""", unsafe_allow_html=True)

    if st.button("🚪 Logout", use_container_width=True, key="do_logout"):
        st.session_state.username = None
        st.session_state.token    = None
        st.rerun()

    st.markdown("<hr style='margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("**⚕️ Health Assessment**")

    with st.form("hf"):
        age   = st.slider("Age",                  1,   100, 30)
        preg  = st.number_input("Pregnancies",    0,    20,  0)
        gluc  = st.slider("Glucose (mg/dL)",     50,   300,100)
        bp    = st.slider("Blood Pressure (mmHg)",40,  180, 70)
        skin  = st.slider("Skin Thickness (mm)",  0,   100, 20)
        ins   = st.slider("Insulin (μU/mL)",      0,  1000, 80)
        bmi   = st.slider("BMI",                10.0, 70.0,25.0)
        ped   = st.slider("Diabetes Pedigree",  0.0,   2.5, 0.5)
        go_   = st.form_submit_button("🔍 Assess My Risk", type="primary")

    return go_, age, preg, gluc, bp, skin, ins, bmi, ped


# =============================================================================
# FEATURE TABS
# =============================================================================
def tab_chat():
    st.markdown("### 🩺 AI Diabetes Specialist")
    st.caption("Powered by GPT-4o-mini · Not a substitute for professional advice")

    assistant = get_chat_assistant()
    if assistant.error:
        st.warning(f"⚠️ {assistant.error}")

    # Chat history
    html = "<div class='chat-box'>"
    if not st.session_state.chat_history:
        html += "<p style='text-align:center;color:rgba(255,255,255,.25);padding:28px 0'>No messages yet — ask below!</p>"
    for m in st.session_state.chat_history:
        ts = datetime.datetime.strptime(m["ts"],"%Y%m%d%H%M%S%f").strftime("%H:%M")
        if m["role"] == "user":
            html += f"<div class='msg-user'>{m['text']}<div class='msg-t' style='text-align:right'>{ts}</div></div>"
        else:
            html += f"<div class='msg-ai'><strong>AI Doctor:</strong> {m['text']}<div class='msg-t'>{ts}</div></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    ci, cb = st.columns([5,1])
    with ci:
        q = st.text_input("q", placeholder="Ask a diabetes question…",
                          label_visibility="collapsed", key="chat_q")
    with cb:
        send = st.button("Send 💬", use_container_width=True)

    if send and q and q.strip():
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        st.session_state.chat_history.append({"role":"user","text":q.strip(),"ts":ts})
        ctx = ""
        if st.session_state.health_history:
            last = st.session_state.health_history[-1]
            ctx = f"{last['risk']} risk, BMI {last['bmi']:.1f}, Glucose {last['glucose']}, Age {last['age']}"
        with st.spinner("Thinking…"):
            reply = assistant.ask(q.strip(), ctx)
        st.session_state.chat_history.append({
            "role":"ai","text":reply,
            "ts":datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clr_chat"):
            st.session_state.chat_history = []; st.rerun()


def tab_timeline():
    st.markdown("### 📈 Health Timeline")
    if not st.session_state.health_history:
        st.info("Complete an assessment to see your timeline here.")
        return

    df = pd.DataFrame(st.session_state.health_history)
    df["date"] = pd.to_datetime(df["date"])

    fig = px.scatter(
        df, x="date", y="probability", color="risk",
        size="probability", hover_data=["bmi","glucose","age"],
        title="Diabetes Risk Over Time",
        color_discrete_map=_RISK_COLORS
    )
    _dark_fig(fig)
    fig.update_layout(height=400, hovermode="x unified",
                      yaxis_title="Probability", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Detailed Records"):
        for e in reversed(st.session_state.health_history):
            rc = _RISK_COLORS.get(e["risk"],"#6C63FF")
            st.markdown(f"""
            <div class="hist-card">
                <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">
                    <div>
                        <b>📅 {e['date']}</b><br>
                        <span>Risk: <strong style="color:{rc}">{e['risk']}</strong>
                        &nbsp;·&nbsp; {e['probability']*100:.1f}%</span>
                    </div>
                    <div style="text-align:right">
                        <span>BMI <b>{e['bmi']:.1f}</b> · Glucose <b>{e['glucose']}</b> mg/dL · Age <b>{e['age']}</b></span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


def tab_meds():
    st.markdown("### 💊 Medication Planner")

    with st.form("mf"):
        st.markdown("**Add Medication**")
        c1,c2 = st.columns(2)
        with c1: name   = st.text_input("Name",   placeholder="e.g. Metformin")
        with c2: dose   = st.text_input("Dosage", placeholder="e.g. 500 mg")
        c1,c2 = st.columns(2)
        with c1: freq  = st.selectbox("Frequency",["Once daily","Twice daily","Three times daily","Weekly"])
        with c2: times = st.multiselect("Time of Day",["Morning","Afternoon","Evening","Night"])
        sd    = st.date_input("Start Date", datetime.date.today())
        notes = st.text_area("Notes (optional)", height=70)
        if st.form_submit_button("➕ Add"):
            if name and dose and times:
                st.session_state.medications.append(
                    {"name":name,"dose":dose,"freq":freq,"times":times,
                     "start":sd.strftime("%Y-%m-%d"),"notes":notes})
                st.success(f"✅ {name} added."); st.rerun()
            else:
                st.error("Name, Dosage and at least one Time are required.")

    st.markdown("#### Your Medications")
    if not st.session_state.medications:
        st.info("No medications yet.")
    else:
        pal = ["#6C63FF","#FF6B6B","#3fb950","#d29922","#4D44DB"]
        for i,m in enumerate(st.session_state.medications):
            ci,cd = st.columns([6,1])
            with ci:
                nt = f"<p>📝 {m['notes']}</p>" if m.get("notes") else ""
                color = pal[i % len(pal)]
                st.markdown(f"""
                <div class="med-card" style="border-left-color:{color}">
                    <h4>💊 {m['name']} <span style="color:{color};font-size:.82rem">({m['dose']})</span></h4>
                    <p>🔁 {m['freq']} &nbsp;·&nbsp; ⏰ {', '.join(m['times'])} &nbsp;·&nbsp; 📅 Since {m['start']}</p>
                    {nt}
                </div>""", unsafe_allow_html=True)
            with cd:
                st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
                if st.button("❌", key=f"del_{i}_{m['name']}"):
                    st.session_state.medications.pop(i); st.rerun()

    st.markdown("#### 📅 Weekly Schedule")
    if not st.session_state.medications:
        st.info("Add medications above to see the calendar.")
        return
    events = []
    pal = ["#6C63FF","#FF6B6B","#3fb950","#d29922","#4D44DB"]
    for i,m in enumerate(st.session_state.medications):
        for period in m["times"]:
            s = _time_for_period(period)
            e = _time_for_period(period, end=True)
            for offset in range(7):
                d = (datetime.date.today()+datetime.timedelta(days=offset)).isoformat()
                events.append({"title":f"{m['name']} ({m['dose']})","color":pal[i%len(pal)],
                               "start":f"{d}T{s}","end":f"{d}T{e}"})
    calendar(events=events, options={
        "headerToolbar":{"left":"today prev,next","center":"title","right":"timeGridWeek,dayGridMonth"},
        "initialView":"timeGridWeek","slotMinTime":"06:00:00","slotMaxTime":"23:00:00",
        "height":"auto","aspectRatio":2,
    }, key="med_cal")


# =============================================================================
# MAIN CONTENT — logged in
# =============================================================================
def page_app(submitted_data):
    # Unpack sidebar form data
    submitted, age, preg, gluc, bp, skin, ins, bmi, ped = submitted_data

    # Header
    username = st.session_state.username
    st.markdown(f"""
    <div class="app-hdr fade-in">
        <div class="app-hdr-icon">🏥</div>
        <div class="app-hdr-text">
            <h1>GlucoCheck Pro+</h1>
            <p>Welcome back, <strong>{username}</strong> · Advanced Diabetes Management with AI Assistance</p>
        </div>
    </div>""", unsafe_allow_html=True)

    if submitted:
        if ML_MODEL is None or ML_SCALER is None:
            st.error("Model not loaded — cannot assess risk."); return

        arr  = np.array([[preg, gluc, bp, skin, ins, bmi, ped, age]])
        prob = ML_MODEL.predict_proba(ML_SCALER.transform(arr))[0][1]
        rec  = _risk(prob, bmi, age)

        entry = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "probability": float(prob), "risk": rec["risk"],
                 "bmi": float(bmi), "glucose": float(gluc), "age": int(age)}
        st.session_state.health_history.append(entry)

        # Download button
        pdf = _pdf(prob, rec, age, bmi, gluc, bp, skin, ins, ped, preg)
        st.download_button("📥 Download PDF Report", data=pdf,
                           file_name=f"glucocheck_{datetime.date.today()}.pdf",
                           mime="application/pdf")

        # Risk summary cards
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="risk-card" style="border-left-color:{rec['color']}">
                <p class="rc-title">Risk Level <span class="rc-icon">{rec['icon']}</span></p>
                <p class="rc-val" style="color:{rec['color']}">{rec['risk']}</p></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="risk-card" style="border-left-color:{rec['color']}">
                <p class="rc-title">Probability <span class="rc-icon">📊</span></p>
                <p class="rc-val">{prob*100:.1f}%</p></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="risk-card" style="border-left-color:#6C63FF">
                <p class="rc-title">BMI Category <span class="rc-icon">⚖️</span></p>
                <p class="rc-val">{rec['bmi_cat']}</p></div>""",
                unsafe_allow_html=True)

        st.markdown("---")
        ga,gb = st.columns([1,2])
        with ga:
            st.plotly_chart(_gauge(prob,rec["color"]), use_container_width=True)
            with st.expander("📊 Your Risk Summary", expanded=True):
                st.markdown(
                    f"- Level: **{rec['risk']}** · Probability: **{prob*100:.1f}%**\n"
                    f"- Age group: **{rec['age_grp']}** · BMI: **{rec['bmi_cat']}**"
                )
                if prob>.6:   st.warning("Consider consulting an endocrinologist.")
                elif prob>.4: st.info("Lifestyle changes can significantly reduce risk.")
                else:         st.success("Keep up your healthy habits!")

        with gb:
            t1,t2,t3 = st.tabs(["🍽️ Diet","🏋️ Exercise","💊 Supplements"])
            with t1:
                for item in rec["diet"]: st.markdown(f"- {item}")
                st.markdown("**Sample Meal Plan**")
                if rec["risk"] in ["High","Very High"]:
                    st.markdown("- **Breakfast**: Veggie omelette + avocado\n"
                                "- **Lunch**: Grilled salmon + quinoa\n"
                                "- **Dinner**: Chicken stir-fry + vegetables")
                else:
                    st.markdown("- **Breakfast**: Oatmeal + berries + nuts\n"
                                "- **Lunch**: Whole-grain wrap + lean protein\n"
                                "- **Dinner**: Baked fish + sweet potato + greens")
            with t2:
                for item in rec["exercise"]: st.markdown(f"- {item}")
                dcols = st.columns(7)
                for i,d in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
                    with dcols[i]: st.caption(d); st.write("🏋️" if i%2==0 else "🏃")
            with t3:
                for item in rec["supplements"]: st.markdown(f"- {item}")
                st.info("Always consult your doctor before starting any supplement.")

        # Insights
        st.markdown("---")
        st.markdown("### 📊 Health Insights")
        i1,i2 = st.tabs(["Risk Factors","Glucose vs BMI"])
        with i1:
            if ML_MODEL and hasattr(ML_MODEL,"feature_importances_"):
                fdf = pd.DataFrame({
                    "Factor":["Pregnancies","Glucose","BP","SkinThickness",
                              "Insulin","BMI","Pedigree","Age"],
                    "Importance": ML_MODEL.feature_importances_
                }).sort_values("Importance",ascending=False)
                fig = px.bar(fdf, x="Importance", y="Factor", orientation="h",
                             color="Importance", color_continuous_scale="Purples",
                             title="Feature Importances")
                st.plotly_chart(_dark_fig(fig), use_container_width=True)
            else:
                st.warning("Feature importances not available for this model type.")
        with i2:
            # ── FIX: removed trendline="lowess" — statsmodels not installed ──
            csv = os.path.join(current_dir,"enhanced_diabetes.csv")
            if os.path.exists(csv):
                df2 = pd.read_csv(csv)
                fig = px.scatter(df2, x="Glucose", y="BMI", color="Outcome",
                                 hover_data=["Age"],
                                 title="Glucose vs BMI (0=No Diabetes, 1=Diabetes)",
                                 color_discrete_map={0:"#3fb950",1:"#f85149"})
                # Manual trend lines using numpy polyfit
                for outcome, col in [(0,"#3fb950"),(1,"#f85149")]:
                    sub = df2[df2["Outcome"]==outcome]
                    if len(sub) > 2:
                        z = np.polyfit(sub["Glucose"], sub["BMI"], 1)
                        x_line = np.linspace(sub["Glucose"].min(), sub["Glucose"].max(), 100)
                        y_line = np.poly1d(z)(x_line)
                        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                                 line=dict(color=col, width=2, dash="dot"),
                                                 name=f"Trend ({'No DM' if outcome==0 else 'DM'})"))
                st.plotly_chart(_dark_fig(fig), use_container_width=True)
            else:
                st.warning("enhanced_diabetes.csv not found — place it in the app directory.")

    else:
        # ── Welcome / dashboard state (no submission yet) ──
        st.markdown("""
        <div class="sec-hdr">
            <h2>Your Health Dashboard</h2>
            <p>Fill in your metrics in the panel on the left, then click <strong>Assess My Risk</strong></p>
            <div class="sec-bar"></div>
        </div>""", unsafe_allow_html=True)

        wc = st.columns(3)
        for col,color,icon,title,desc in [
            (wc[0],"#6C63FF","🎯","Risk Assessment","AI-powered score with personalised diet, exercise & supplement plans."),
            (wc[1],"#FF6B6B","💊","Medication Tracker","Track all doses on a smart calendar — never miss a reminder."),
            (wc[2],"#3fb950","📈","Health Timeline","Visualise how your risk evolves over time."),
        ]:
            with col:
                st.markdown(f"""
                <div class="feat-card" style="border-top-color:{color}">
                    <span class="fi">{icon}</span>
                    <h3>{title}</h3><p>{desc}</p>
                </div>""", unsafe_allow_html=True)

    # ── Premium Tabs ──
    st.markdown("---")
    st.markdown("""
    <div class="sec-hdr">
        <h2>✨ Premium Features</h2>
        <p>AI chat, history tracking &amp; medication planning</p>
        <div class="sec-bar"></div>
    </div>""", unsafe_allow_html=True)

    ft = st.tabs(["🩺 AI Doctor Chat","📈 Health Timeline","💊 Medication Planner"])
    with ft[0]: tab_chat()
    with ft[1]: tab_timeline()
    with ft[2]: tab_meds()


# =============================================================================
# LANDING PAGE  (not logged in)
# =============================================================================
def page_landing():
    st.markdown("""
    <div class="hero fade-in">
        <div class="hero-badge">🏥 Advanced Diabetes Management</div>
        <h1>GlucoCheck <span>Pro+</span></h1>
        <p>AI-powered risk assessment, personalised health plans, and smart medication
           tracking — built for people who take their diabetes management seriously.</p>
        <div class="hero-pills">
            <span class="hero-pill">🎯 AI Risk Assessment</span>
            <span class="hero-pill">🩺 GPT-4o Chat</span>
            <span class="hero-pill">💊 Medication Tracker</span>
            <span class="hero-pill">📈 Health Timeline</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-bar">
        <div class="stat"><span class="n">95%</span><span class="l">MODEL ACCURACY</span></div>
        <div class="stat"><span class="n">5</span><span class="l">RISK TIERS</span></div>
        <div class="stat"><span class="n">GPT-4o</span><span class="l">AI CHAT</span></div>
        <div class="stat"><span class="n">PDF</span><span class="l">HEALTH REPORTS</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-hdr">
        <h2>Everything You Need</h2>
        <p>Comprehensive tools to understand and manage your diabetes risk</p>
        <div class="sec-bar"></div>
    </div>""", unsafe_allow_html=True)

    fc = st.columns(4)
    for col,color,icon,title,desc in [
        (fc[0],"#6C63FF","🎯","Risk Assessment","Instant AI-powered diabetes probability with personalised plans."),
        (fc[1],"#FF6B6B","🩺","AI Doctor Chat","GPT-4o powered virtual specialist — ask anything, get fast answers."),
        (fc[2],"#3fb950","💊","Medication Planner","Weekly calendar tracking — never miss a dose again."),
        (fc[3],"#d29922","📈","Health Timeline","Track risk trends and see the impact of lifestyle changes."),
    ]:
        with col:
            st.markdown(f"""
            <div class="feat-card" style="border-top-color:{color}">
                <span class="fi">{icon}</span><h3>{title}</h3><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-hdr">
        <h2>How It Works</h2>
        <p>Three steps to better diabetes management</p>
        <div class="sec-bar"></div>
    </div>""", unsafe_allow_html=True)

    sc = st.columns(3)
    for col,num,title,desc in [
        (sc[0],"01","Create Account",    "Register in seconds. Your data is private and never shared."),
        (sc[1],"02","Enter Your Metrics","Input glucose, BMI, blood pressure and other health vitals."),
        (sc[2],"03","Get Your Report",   "Personalised risk score, recommendations & downloadable PDF."),
    ]:
        with col:
            st.markdown(f"""
            <div class="step-card">
                <span class="step-num">{num}</span><h4>{title}</h4><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Login or Register** using the panel on the left to get started.")


# =============================================================================
# ENTRY POINT  —  2-column layout replaces the native sidebar
# =============================================================================
def main():
    initialize_user_db()

    username = st.session_state.username

    if username:
        # Logged in: narrow left panel + wide content area
        left, right = st.columns([1, 3.8], gap="small")
        with left:
            st.markdown('<div class="left-panel">', unsafe_allow_html=True)
            submitted_data = _panel_app(username)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            page_app(submitted_data)
    else:
        # Logged out: narrow auth panel + landing page
        left, right = st.columns([1, 3.2], gap="small")
        with left:
            st.markdown('<div class="left-panel">', unsafe_allow_html=True)
            _panel_auth()
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            page_landing()


if __name__ == "__main__":
    main()

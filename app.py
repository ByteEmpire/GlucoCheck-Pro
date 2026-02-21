import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.card import card
import time
import sys
import os
import google.generativeai as genai
import datetime
from streamlit_calendar import calendar
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import uuid

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
auth_dir = os.path.join(current_dir, 'auth')
if auth_dir not in sys.path:
    sys.path.append(auth_dir)

# --- Page Config ---
st.set_page_config(
    page_title="GlucoCheck Pro+",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

startup_notices = []

confusion_matrix_path = os.path.join(current_dir, "confusion_matrix.html")
if os.path.exists(confusion_matrix_path):
    try:
        os.remove(confusion_matrix_path)
    except Exception:
        pass

# --- Authentication Imports ---
try:
    from auth_utils import (
        initialize_user_db, hash_password, register_user, verify_user,
        generate_token, verify_token, update_password, delete_user
    )
except ImportError:
    startup_notices.append(("error", "Authentication utilities not found."))
    def initialize_user_db(): pass
    def hash_password(p): return p
    def register_user(u, p): return False
    def verify_user(u, p): return False
    def generate_token(u): return "dummy_token"
    def verify_token(t): return None
    def update_password(u, p): return False
    def delete_user(u): return False

for level, message in startup_notices:
    if level == "success": st.success(message)
    elif level == "warning": st.warning(message)
    else: st.error(message)


# =============================================================================
# GLOBAL CSS
# =============================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #6C63FF;
        --secondary: #4D44DB;
        --bg-dark: #0d1117;
        --bg-card: #161b22;
        --bg-card2: #1c2128;
        --border: rgba(108,99,255,0.2);
        --text: rgba(255,255,255,0.88);
        --muted: rgba(255,255,255,0.5);
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* ── UNIFIED SCROLL ── */
    html, body {
        overflow-y: auto !important;
        height: auto !important;
    }
    [data-testid="stAppViewContainer"] {
        overflow: visible !important;
        height: auto !important;
    }
    section[data-testid="stMain"] {
        overflow: visible !important;
        height: auto !important;
    }
    [data-testid="stSidebar"] {
        position: sticky !important;
        top: 0 !important;
        height: 100vh !important;
        overflow-y: auto !important;
        background: linear-gradient(180deg, #0d1117 0%, #161b22 60%, #1c2128 100%) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* ── Block container ── */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* ── Dark background everywhere ── */
    .stApp { background: var(--bg-dark) !important; }
    [data-testid="stMain"] > div { background: var(--bg-dark) !important; }

    /* ── All text dark-theme ── */
    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] label,
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] li {
        color: var(--text) !important;
    }

    hr { border-color: var(--border) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #4D44DB) !important;
        color: white !important;
        border: none !important;
        padding: 10px 26px !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(108,99,255,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108,99,255,0.5) !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tab"] { color: var(--muted) !important; font-weight: 500 !important; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--primary) !important; border-bottom-color: var(--primary) !important; }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary { color: var(--text) !important; }

    /* ── Forms ── */
    [data-testid="stForm"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 20px !important;
    }
    [data-testid="stForm"] input,
    [data-testid="stForm"] textarea {
        background: rgba(255,255,255,0.05) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        background: rgba(108,99,255,0.1) !important;
        border: 1px solid rgba(108,99,255,0.25) !important;
        border-radius: 10px !important;
    }

    /* ═══════════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════════ */
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(108,99,255,0.4) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    [data-testid="stSidebar"] input::placeholder { color: rgba(255,255,255,0.3) !important; }
    [data-testid="stSidebar"] input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 0.95rem !important;
        margin-top: 6px !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(108,99,255,0.2) !important; margin: 14px 0 !important; }

    /* ═══════════════════════════════════════
       CUSTOM COMPONENTS
    ═══════════════════════════════════════ */

    /* Sidebar brand card */
    .sidebar-brand {
        background: rgba(108,99,255,0.12); border: 1px solid rgba(108,99,255,0.3);
        border-radius: 14px; padding: 18px 14px 14px; text-align: center; margin-bottom: 18px;
    }
    .sidebar-brand .brand-icon { font-size: 2.2rem; display: block; margin-bottom: 5px; }
    .sidebar-brand .brand-name { font-size: 1.15rem; font-weight: 700; color: white; display: block; margin-bottom: 2px; }
    .sidebar-brand .brand-tag  { font-size: 0.7rem; color: rgba(255,255,255,0.4); letter-spacing: 1px; text-transform: uppercase; }

    /* Auth tab switcher */
    .auth-tab-row { display: flex; gap: 4px; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px; margin-bottom: 18px; }
    .auth-tab { flex: 1; text-align: center; padding: 8px 0; border-radius: 9px; font-size: 0.85rem; font-weight: 600; color: rgba(255,255,255,0.45); }
    .auth-tab.active { background: linear-gradient(135deg, #6C63FF, #4D44DB); color: white; box-shadow: 0 3px 10px rgba(108,99,255,0.4); }
    .auth-section-label { font-size: 0.76rem; font-weight: 600; color: rgba(255,255,255,0.4) !important; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px; display: block; }

    /* User profile in sidebar */
    .sidebar-user {
        background: rgba(108,99,255,0.1); border: 1px solid rgba(108,99,255,0.25);
        border-radius: 12px; padding: 14px; text-align: center; margin-bottom: 16px;
    }
    .sidebar-user .user-avatar { font-size: 2.4rem; display: block; margin-bottom: 4px; }
    .sidebar-user .user-name   { font-size: 1rem; font-weight: 600; color: white; }
    .sidebar-user .user-tag    { font-size: 0.72rem; color: rgba(255,255,255,0.4); }

    /* App header (logged in) */
    .app-header {
        background: linear-gradient(135deg, #0d1117, #161b22, #1c2128);
        border-radius: 16px; padding: 26px 32px; margin-bottom: 22px;
        border: 1px solid var(--border); display: flex; align-items: center; gap: 18px;
    }
    .app-header-text h1 { font-size: 1.75rem; font-weight: 700; color: white; margin: 0 0 4px; }
    .app-header-text p  { font-size: 0.86rem; color: var(--muted); margin: 0; }

    /* Risk cards */
    .risk-card {
        background: var(--bg-card); border-radius: 12px; padding: 18px;
        border-left: 5px solid var(--primary); box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .risk-card h3 { margin-top: 0; margin-bottom: 8px; }
    .risk-card p  { font-size: 1.45rem; font-weight: 600; margin: 0; color: white; }

    /* Medication card */
    .med-card {
        background: var(--bg-card); border-left: 4px solid #6C63FF;
        border-radius: 12px; padding: 15px 18px; margin-bottom: 10px;
        border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
    }
    .med-card h4 { color: white; margin: 0 0 7px; font-size: 0.98rem; }
    .med-card p  { color: var(--muted); margin: 2px 0; font-size: 0.84rem; }

    /* History card */
    .history-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
    }
    .history-card b   { color: white; }
    .history-card span { color: var(--muted); font-size: 0.87rem; }

    /* Chat */
    .chat-wrap {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 14px; padding: 16px; margin-bottom: 14px;
        max-height: 420px; overflow-y: auto;
    }
    .user-message {
        background: linear-gradient(135deg, #6C63FF, #4D44DB); color: white;
        border-radius: 18px 18px 0 18px; padding: 11px 15px; margin: 8px 0 8px auto;
        max-width: 80%; word-wrap: break-word; box-shadow: 0 2px 8px rgba(108,99,255,0.3); display: block;
    }
    .ai-message {
        background: var(--bg-card2); color: var(--text);
        border-radius: 18px 18px 18px 0; padding: 11px 15px; margin: 8px auto 8px 0;
        max-width: 80%; word-wrap: break-word; border: 1px solid var(--border); display: block;
    }
    .msg-time { font-size: 0.67em; opacity: 0.5; margin-top: 4px; }

    /* Hero */
    .hero-section {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2128 100%);
        border-radius: 20px; padding: 64px 50px 56px; margin-bottom: 26px; text-align: center;
        position: relative; overflow: hidden; border: 1px solid var(--border);
    }
    .hero-section::before {
        content: ''; position: absolute; top: -40%; left: -20%; width: 160%; height: 160%;
        background: radial-gradient(circle at 60% 40%, rgba(108,99,255,0.14) 0%, transparent 55%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block; background: rgba(108,99,255,0.2); border: 1px solid rgba(108,99,255,0.4);
        color: #c4c0ff; padding: 5px 18px; border-radius: 50px;
        font-size: 0.8rem; font-weight: 500; margin-bottom: 16px; letter-spacing: 0.8px; text-transform: uppercase;
    }
    .hero-title { font-size: 3rem; font-weight: 700; color: #fff; margin: 0 0 14px; line-height: 1.15; }
    .hero-title span { background: linear-gradient(90deg,#a78bfa,#818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { font-size: 1.05rem; color: rgba(255,255,255,0.65); margin: 0 auto 30px; font-weight: 300; max-width: 540px; }
    .hero-features { display: flex; justify-content: center; gap: 14px; flex-wrap: wrap; margin-top: 14px; }
    .hero-feature-pill {
        background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
        color: rgba(255,255,255,0.8); padding: 7px 18px; border-radius: 50px; font-size: 0.85rem; font-weight: 500;
    }

    /* Stats bar */
    .stats-bar {
        background: linear-gradient(135deg, #6C63FF, #4D44DB);
        border-radius: 16px; padding: 26px 20px; margin: 0 0 26px;
        display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 16px;
    }
    .stat-item { text-align: center; }
    .stat-number { font-size: 1.9rem; font-weight: 700; color: #fff; display: block; }
    .stat-label  { font-size: 0.75rem; color: rgba(255,255,255,0.65); letter-spacing: 0.5px; }

    /* Feature cards */
    .feature-card {
        background: var(--bg-card); border-radius: 16px; padding: 24px 18px;
        border-top: 4px solid; border-left: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
        text-align: center; height: 100%; transition: transform 0.25s, box-shadow 0.25s;
    }
    .feature-card:hover { transform: translateY(-4px); box-shadow: 0 14px 36px rgba(0,0,0,0.45); }
    .feature-icon { font-size: 2.2rem; margin-bottom: 10px; display: block; }
    .feature-card h3 { font-size: 1rem; font-weight: 600; color: white; margin-bottom: 7px; }
    .feature-card p  { font-size: 0.85rem; color: var(--muted); line-height: 1.5; margin: 0; }

    /* Step cards */
    .step-card { background: var(--bg-card); border-radius: 14px; padding: 22px 16px; text-align: center; border: 1px solid rgba(108,99,255,0.15); }
    .step-number { font-size: 1.9rem; font-weight: 700; color: #6C63FF; display: block; margin-bottom: 5px; }
    .step-card h4 { font-size: 0.94rem; font-weight: 600; color: white; margin-bottom: 5px; }
    .step-card p  { font-size: 0.82rem; color: var(--muted); margin: 0; }

    /* Section header */
    .section-header { text-align: center; margin: 32px 0 20px; }
    .section-header h2 { font-size: 1.6rem; font-weight: 700; color: white; margin-bottom: 5px; }
    .section-header p  { color: var(--muted); font-size: 0.9rem; margin: 0; }
    .section-divider { width: 44px; height: 4px; background: linear-gradient(90deg,#6C63FF,#4D44DB); border-radius: 2px; margin: 7px auto 0; }

    @keyframes fadeInUp { from {opacity:0;transform:translateY(14px);} to {opacity:1;transform:translateY(0);} }
    .fade-in { animation: fadeInUp 0.45s ease-out; }
    </style>
    """, unsafe_allow_html=True)

inject_css()


# =============================================================================
# AI DOCTOR ASSISTANT  —  fixed model discovery
# =============================================================================
class AIDoctorAssistant:
    def __init__(self):
        self.model       = None
        self.init_error  = None
        self.system_prompt = (
            "You are a friendly, knowledgeable diabetes specialist AI assistant. "
            "Provide accurate, evidence-based information about diabetes prevention, "
            "management, diet, exercise, and medication. Always advise users to consult "
            "a qualified healthcare professional for personal medical decisions."
        )

        api_key = None
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.init_error = "Missing GEMINI_API_KEY in Streamlit secrets or environment."
            return

        try:
            genai.configure(api_key=api_key)

            # Prefer newer, stable models first — try both with and without prefix
            candidates = [
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "models/gemini-2.0-flash",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-flash-latest",
                "models/gemini-pro",
            ]

            # Try to get available model names; fall back gracefully if listing fails
            available = set()
            try:
                for m in genai.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        available.add(m.name)
                        available.add(m.name.replace("models/", ""))
            except Exception:
                pass  # If listing fails, try all candidates anyway

            for cand in candidates:
                short = cand.replace("models/", "")
                if available and cand not in available and short not in available:
                    continue
                try:
                    self.model = genai.GenerativeModel(cand)
                    break
                except Exception:
                    self.model = None
                    continue

            if self.model is None:
                self.init_error = "No supported Gemini model found. Check your API key / quota."

        except Exception as e:
            self.init_error = f"Failed to initialise Gemini: {e}"

    def generate_response(self, user_query, context=""):
        if not self.model:
            return "AI service unavailable. Please check configuration."
        try:
            prompt = f"{self.system_prompt}\n\nContext: {context}\n\nQuestion: {user_query}"
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7, max_output_tokens=512, top_p=0.9
                )
            )
            return response.text
        except Exception as e:
            return f"I'm having trouble responding right now ({e}). Please try again."


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    for k, v in {
        'token': None, 'username': None,
        'chat_history': [], 'medications': [],
        'health_history': [], 'last_assessment': None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model():
    model = scaler = None
    try:
        mp = os.path.join(current_dir, "model.pkl")
        sp = os.path.join(current_dir, "scaler.pkl")
        if os.path.exists(mp): model  = joblib.load(mp)
        if os.path.exists(sp): scaler = joblib.load(sp)
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
    return model, scaler

model, scaler = load_model()


# =============================================================================
# HELPERS
# =============================================================================
def get_personalized_recommendations(probability, bmi, age):
    ag  = "child/teen" if age < 18 else ("senior" if age > 60 else "adult")
    bmc = ("Underweight" if bmi < 18.5 else "Normal Weight" if bmi < 25
           else "Overweight" if bmi < 30 else "Obese")

    tiers = [
        (0.2, "Very Low", "#4CAF50", "😊",
         ["Balanced diet, whole foods.", "Limit processed sugars.", "Stay hydrated."],
         ["Brisk walking 30 min, 5×/week.", "Strength training 2-3×/week."],
         ["Multivitamin.", "Omega-3 fatty acids."]),
        (0.4, "Low", "#8BC34A", "🙂",
         ["Portion control.", "Increase fibre intake.", "Reduce sugary drinks."],
         ["150 min moderate exercise/week.", "Cycling, swimming, dancing."],
         ["Vitamin D."]),
        (0.6, "Moderate", "#FFC107", "😐",
         ["Limit refined carbs.", "Lean proteins + healthy fats.", "Consult a nutritionist."],
         ["200–250 min moderate/week.", "Cardio + strength training."],
         ["Chromium picolinate.", "Magnesium."]),
        (0.8, "High", "#FF9800", "😟",
         ["Low-GI diet.", "Eliminate sugary snacks.", "Work with a dietitian."],
         ["250–300 min moderate or 150 min vigorous/week.", "Monitor blood sugar."],
         ["Berberine (consult doctor).", "Alpha-lipoic acid."]),
        (1.1, "Very High", "#F44336", "❗",
         ["Medical nutrition therapy.", "Avoid processed foods.", "Regular blood sugar monitoring."],
         ["Daily activity, even short walks.", "Supervised programme."],
         ["Supplements under medical supervision.", "Coenzyme Q10."]),
    ]
    for thresh, risk, color, icon, diet, exercise, supps in tiers:
        if probability < thresh:
            return dict(risk_level=risk, color=color, icon=icon,
                        diet=diet, exercise=exercise, supplements=supps,
                        bmi_category=bmc, age_group=ag)


def create_gauge(probability, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        domain={'x':[0,1],'y':[0,1]},
        title={'text':"Diabetes Risk Score (%)","font":{"size":18,"color":"white"}},
        number={'font':{'color':'white'}},
        gauge={
            'axis': {'range':[None,100],'tickcolor':'#aaa'},
            'bar':  {'color': color},
            'bgcolor': "#161b22", 'borderwidth':1, 'bordercolor':"#30363d",
            'steps': [
                {'range':[0, 20],'color':"#0d2818"},
                {'range':[20,40],'color':"#1a3a1a"},
                {'range':[40,60],'color':"#3a2e00"},
                {'range':[60,80],'color':"#3a1a00"},
                {'range':[80,100],'color':"#3a0000"},
            ],
            'threshold': {'line':{'color':"red",'width':4},'thickness':0.75,'value':probability*100}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=44,b=10),
                      paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="white")
    return fig


def risk_card(title, value, color, icon):
    st.markdown(f"""
    <div class="risk-card" style="border-left-color:{color};">
        <h3 style="color:{color};margin-top:0;margin-bottom:8px;">{icon} {title}</h3>
        <p>{value}</p>
    </div>""", unsafe_allow_html=True)


def generate_pdf_report(probability, rec, age, bmi, glucose, bp, skin, insulin, pedigree, preg):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold",22); c.setFillColorRGB(0.42,0.39,1.0)
    c.drawString(50, h-50, "GlucoCheck Pro+ Health Report")
    c.setFont("Helvetica",11); c.setFillColorRGB(0.5,0.5,0.5)
    c.drawString(50, h-68, f"Date: {datetime.date.today().strftime('%Y-%m-%d')}")
    y = h-115
    c.setFont("Helvetica-Bold",13); c.setFillColorRGB(0,0,0)
    c.drawString(50, y, "Patient Information:")
    c.setFont("Helvetica",11)
    for line in [f"Age: {age} yrs", f"BMI: {bmi:.1f} ({rec['bmi_category']})",
                 f"Glucose: {glucose} mg/dL", f"BP: {bp} mmHg",
                 f"Skin Thickness: {skin} mm", f"Insulin: {insulin} μU/mL",
                 f"Pedigree: {pedigree:.3f}", f"Pregnancies: {preg}"]:
        y -= 16; c.drawString(68, y, line)
    y -= 28; c.setFont("Helvetica-Bold",13); c.drawString(50, y, "Risk Assessment:")
    c.setFont("Helvetica",11); y -= 16; c.drawString(68, y, f"Probability: {probability*100:.1f}%")
    y -= 16; c.drawString(68, y, f"Risk Level: {rec['risk_level']}")
    y -= 28; c.setFont("Helvetica-Bold",13); c.drawString(50, y, "Recommendations:")
    for section, items in [("Diet",rec["diet"]),("Exercise",rec["exercise"]),("Supplements",rec["supplements"])]:
        y -= 20; c.setFont("Helvetica-Bold",11); c.drawString(68, y, f"{section}:")
        c.setFont("Helvetica",11)
        for item in items:
            y -= 14; c.drawString(80, y, f"• {item}")
    c.setFont("Helvetica-Oblique",9); c.setFillColorRGB(0.5,0.5,0.5)
    c.drawString(50,45,"Disclaimer: Risk assessment only. Always consult a qualified healthcare professional.")
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()


def get_time_for_period(period, end=False):
    t = {"Morning":("08:00:00","09:00:00"),"Afternoon":("12:00:00","13:00:00"),
         "Evening":("18:00:00","19:00:00"),"Night":("21:00:00","22:00:00")}
    return t[period][1 if end else 0]


# =============================================================================
# AI DOCTOR CHAT
# =============================================================================
def ai_doctor_chat():
    st.markdown("### 🩺 AI Diabetes Specialist")
    st.caption("Evidence-based answers — not a substitute for professional medical advice")

    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = AIDoctorAssistant()
    assistant = st.session_state.ai_assistant

    if assistant.init_error:
        st.warning(f"⚠️ {assistant.init_error}")

    chat_html = "<div class='chat-wrap'>"
    if not st.session_state.chat_history:
        chat_html += "<p style='text-align:center;color:rgba(255,255,255,0.28);padding:28px 0;'>No messages yet — ask your first question below!</p>"
    for msg in st.session_state.chat_history:
        ts = datetime.datetime.strptime(msg['timestamp'],"%Y%m%d%H%M%S%f").strftime("%H:%M")
        if msg['role'] == 'user':
            chat_html += f"<div class='user-message'>{msg['content']}<div class='msg-time' style='text-align:right'>{ts}</div></div>"
        else:
            chat_html += f"<div class='ai-message'><strong>AI Doctor:</strong> {msg['content']}<div class='msg-time'>{ts}</div></div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    c1, c2 = st.columns([5,1])
    with c1:
        user_query = st.text_input("q", placeholder="Ask a diabetes-related question…",
                                   label_visibility="collapsed", key="chat_text_input")
    with c2:
        send = st.button("Send 💬", use_container_width=True)

    if send and user_query and user_query.strip():
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        st.session_state.chat_history.append({"role":"user","content":user_query.strip(),"timestamp":ts})
        ctx = ""
        if st.session_state.health_history:
            last = st.session_state.health_history[-1]
            ctx = f"{last['risk_level']} risk, BMI {last['bmi']}, Glucose {last['glucose']}, Age {last['age']}"
        with st.spinner("AI Doctor is thinking…"):
            reply = assistant.generate_response(user_query.strip(), ctx)
        st.session_state.chat_history.append({
            "role":"assistant","content":reply,
            "timestamp":datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# =============================================================================
# HEALTH TIMELINE  — stylable_container removed entirely
# =============================================================================
def display_health_timeline():
    st.markdown("### 📈 Your Health Timeline")
    if not st.session_state.health_history:
        st.info("No assessments yet. Complete an assessment to see your timeline.")
        return

    df = pd.DataFrame(st.session_state.health_history)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    fig = px.scatter(
        df, x='date', y='probability', color='risk_level',
        size='probability', hover_data=['bmi','glucose','age'],
        title="Your Diabetes Risk Over Time",
        color_discrete_map={"Very Low":"#4CAF50","Low":"#8BC34A","Moderate":"#FFC107",
                            "High":"#FF9800","Very High":"#F44336"}
    )
    fig.update_layout(
        yaxis_title="Probability", xaxis_title="Date",
        hovermode="x unified", height=420,
        paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📝 Detailed History"):
        rc_map = {"Very Low":"#4CAF50","Low":"#8BC34A","Moderate":"#FFC107",
                  "High":"#FF9800","Very High":"#F44336"}
        for entry in reversed(st.session_state.health_history):
            rc = rc_map.get(entry['risk_level'], "#6C63FF")
            st.markdown(f"""
            <div class="history-card">
                <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                    <div>
                        <b>📅 {entry['date']}</b><br>
                        <span>Risk: <strong style="color:{rc}">{entry['risk_level']}</strong>
                        &nbsp;|&nbsp; Probability: <strong style="color:{rc}">{entry['probability']*100:.1f}%</strong></span>
                    </div>
                    <div style="text-align:right;">
                        <span>BMI: <b>{entry['bmi']:.1f}</b> &nbsp;|&nbsp;
                        Glucose: <b>{entry['glucose']}</b> mg/dL &nbsp;|&nbsp;
                        Age: <b>{entry['age']}</b></span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)


# =============================================================================
# MEDICATION PLANNER  — stylable_container removed entirely
# =============================================================================
def medication_planner():
    st.markdown("### 💊 Medication & Reminder Planner")

    with st.form("add_medication"):
        st.markdown("**Add New Medication**")
        c1, c2 = st.columns(2)
        with c1: name   = st.text_input("Medication Name", placeholder="e.g. Metformin")
        with c2: dosage = st.text_input("Dosage",          placeholder="e.g. 500mg")

        c1, c2 = st.columns(2)
        with c1: frequency   = st.selectbox("Frequency", ["Once daily","Twice daily","Three times daily","Weekly"])
        with c2: time_of_day = st.multiselect("Time of Day", ["Morning","Afternoon","Evening","Night"])

        start_date = st.date_input("Start Date", datetime.date.today())
        notes      = st.text_area("Notes (optional)")

        if st.form_submit_button("➕ Add Medication"):
            if name and dosage and time_of_day:
                st.session_state.medications.append({
                    "name": name, "dosage": dosage, "frequency": frequency,
                    "times": time_of_day,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "notes": notes,
                })
                st.success(f"✅ {name} added!")
                st.rerun()
            else:
                st.error("Please fill in Name, Dosage, and at least one Time.")

    st.markdown("#### Your Medications")
    if not st.session_state.medications:
        st.info("No medications added yet.")
    else:
        for i, med in enumerate(st.session_state.medications):
            ci, cd = st.columns([5,1])
            with ci:
                notes_html = f"<p>📝 {med['notes']}</p>" if med.get('notes') else ""
                st.markdown(f"""
                <div class="med-card">
                    <h4>💊 {med['name']} <span style="color:#6C63FF;font-size:0.82rem;">({med['dosage']})</span></h4>
                    <p>🔁 {med['frequency']} &nbsp;|&nbsp; ⏰ {', '.join(med['times'])} &nbsp;|&nbsp; 📅 Since {med['start_date']}</p>
                    {notes_html}
                </div>""", unsafe_allow_html=True)
            with cd:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("❌", key=f"del_{i}_{med['name']}"):
                    st.session_state.medications.pop(i)
                    st.rerun()

    st.markdown("#### 📅 Weekly Schedule")
    if st.session_state.medications:
        events = []
        pal = ["#6C63FF","#FF6B6B","#4CAF50","#FFC107","#4D44DB"]
        for i, med in enumerate(st.session_state.medications):
            for period in med['times']:
                s = get_time_for_period(period, end=False)
                e = get_time_for_period(period, end=True)
                for offset in range(7):
                    d = (datetime.date.today() + datetime.timedelta(days=offset)).isoformat()
                    events.append({"title":f"{med['name']} ({med['dosage']})",
                                   "color":pal[i%len(pal)],
                                   "start":f"{d}T{s}","end":f"{d}T{e}"})
        calendar(events=events, options={
            "headerToolbar":{"left":"today prev,next","center":"title",
                             "right":"dayGridMonth,timeGridWeek,timeGridDay"},
            "initialView":"timeGridWeek",
            "slotMinTime":"06:00:00","slotMaxTime":"23:00:00",
            "height":"auto","aspectRatio":2,
        }, key="med_calendar")
    else:
        st.info("Add medications above to see the schedule calendar.")


# =============================================================================
# MAIN APP CONTENT (logged in)
# =============================================================================
def app_main_content():
    # ── Styled header ──
    st.markdown(f"""
    <div class="app-header fade-in">
        <div style="font-size:3rem;line-height:1;">🏥</div>
        <div class="app-header-text">
            <h1>GlucoCheck Pro+</h1>
            <p>Welcome back, <strong>{st.session_state.username}</strong>
               · Advanced Diabetes Management with AI Assistance</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-user">
            <span class="user-avatar">👤</span>
            <span class="user-name">{st.session_state.username}</span>
            <span class="user-tag">Active session</span>
        </div>""", unsafe_allow_html=True)

        if st.button("🚪 Logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()

        st.markdown("---")
        st.markdown("**⚕️ Health Assessment**")
        with st.form("input_form"):
            age            = st.slider("Age", 1, 100, 30)
            pregnancies    = st.number_input("Pregnancies", 0, 20, 0)
            glucose        = st.slider("Glucose (mg/dL)", 50, 300, 100)
            bp             = st.slider("Blood Pressure (mmHg)", 40, 180, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
            insulin        = st.slider("Insulin (μU/mL)", 0, 1000, 80)
            bmi            = st.slider("BMI", 10.0, 70.0, 25.0)
            pedigree       = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5)
            submitted      = st.form_submit_button("🔍 Assess My Risk", type="primary")

    # ── Risk Assessment ──
    if submitted:
        if model is None or scaler is None:
            st.error("Cannot assess: model/scaler not loaded.")
            return

        arr      = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, pedigree, age]])
        prob     = model.predict_proba(scaler.transform(arr))[0][1]
        rec      = get_personalized_recommendations(prob, bmi, age)

        entry = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "probability": float(prob), "risk_level": rec["risk_level"],
                 "bmi": float(bmi), "glucose": float(glucose), "age": int(age), "bp": float(bp)}
        st.session_state.health_history.append(entry)
        st.session_state.last_assessment = entry

        st.success("✅ Analysis complete — here's your personalised report:")
        pdf = generate_pdf_report(prob, rec, age, bmi, glucose, bp, skin_thickness, insulin, pedigree, pregnancies)
        st.download_button("📥 Download PDF Report", data=pdf,
                           file_name=f"glucocheck_{datetime.date.today()}.pdf",
                           mime="application/pdf")

        c1, c2, c3 = st.columns(3)
        with c1: risk_card("Risk Level",   rec["risk_level"],       rec["color"], rec["icon"])
        with c2: risk_card("Probability",  f"{prob*100:.1f}%",      rec["color"], "📊")
        with c3: risk_card("BMI Category", rec["bmi_category"],     "#6C63FF",    "⚖️")

        st.markdown("---")
        ga, gb = st.columns([1,2])
        with ga:
            st.plotly_chart(create_gauge(prob, rec["color"]), use_container_width=True)
            with st.expander("📊 Understanding Your Risk", expanded=True):
                st.markdown(f"- Risk level: **{rec['risk_level']}**\n- Age group: **{rec['age_group'].title()}**\n- BMI: **{rec['bmi_category']}**")
                if prob > 0.6:   st.warning("Consider consulting an endocrinologist.")
                elif prob > 0.4: st.info("Lifestyle changes can significantly reduce risk.")
                else:            st.success("Keep up your healthy habits!")
        with gb:
            t1,t2,t3 = st.tabs(["🍽️ Diet","🏋️ Exercise","💊 Supplements"])
            with t1:
                st.subheader("Nutrition Guide")
                for item in rec["diet"]: st.markdown(f"- {item}")
                st.markdown("**Sample Meal Plan**")
                if rec["risk_level"] in ["High","Very High"]:
                    st.markdown("- **Breakfast**: Veggie omelette + avocado\n- **Lunch**: Salmon + quinoa\n- **Dinner**: Chicken stir-fry")
                else:
                    st.markdown("- **Breakfast**: Oatmeal + berries\n- **Lunch**: Whole-grain wrap\n- **Dinner**: Baked fish + sweet potato")
            with t2:
                st.subheader("Fitness Recommendations")
                for item in rec["exercise"]: st.markdown(f"- {item}")
                day_cols = st.columns(7)
                for i, d in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
                    with day_cols[i]: st.caption(d); st.write("🏋️" if i%2==0 else "🏃")
            with t3:
                st.subheader("Suggested Supplements")
                for item in rec["supplements"]: st.markdown(f"- {item}")
                st.info("Always consult your doctor before starting supplements.")

        st.markdown("---")
        st.markdown("### 📊 Health Insights")
        i1, i2 = st.tabs(["Risk Factors","Glucose Analysis"])
        with i1:
            if model and hasattr(model, "feature_importances_"):
                fdf = pd.DataFrame({
                    "Factor":['Pregnancies','Glucose','BP','SkinThickness','Insulin','BMI','Pedigree','Age'],
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
                fig = px.bar(fdf, x="Importance", y="Factor", orientation="h",
                             color="Importance", color_continuous_scale="Purples",
                             title="What Impacts Your Risk Most?")
                fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance not available for this model.")
        with i2:
            csv = os.path.join(current_dir, "enhanced_diabetes.csv")
            if os.path.exists(csv):
                df2 = pd.read_csv(csv)
                fig = px.scatter(df2, x="Glucose", y="BMI", color="Outcome",
                                 hover_data=["Age"], title="Glucose vs BMI",
                                 color_discrete_map={0:"#4CAF50",1:"#F44336"}, trendline="lowess")
                fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("enhanced_diabetes.csv not found.")

    else:
        # Welcome screen (no submission yet)
        st.markdown("""
        <div class="section-header">
            <h2>Your Health Dashboard</h2>
            <p>Fill in your metrics in the sidebar and click <strong>Assess My Risk</strong></p>
            <div class="section-divider"></div>
        </div>""", unsafe_allow_html=True)
        wc = st.columns(3)
        for col, icon, title, desc, color in [
            (wc[0],"🎯","Risk Assessment","AI-powered diabetes risk score with personalised recommendations.","#6C63FF"),
            (wc[1],"💊","Medication Tracker","Track doses and never miss a medication reminder.","#FF6B6B"),
            (wc[2],"📈","Health Timeline","See how your risk evolves over time.","#4CAF50"),
        ]:
            with col:
                st.markdown(f"""
                <div class="feature-card" style="border-top-color:{color};">
                    <span class="feature-icon">{icon}</span>
                    <h3>{title}</h3><p>{desc}</p>
                </div>""", unsafe_allow_html=True)

    # ── Premium Feature Tabs ──
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h2>✨ Premium Features</h2>
        <p>AI chat, health history &amp; medication planning</p>
        <div class="section-divider"></div>
    </div>""", unsafe_allow_html=True)

    ft = st.tabs(["🩺 AI Doctor Chat","📈 Health Timeline","💊 Medication Planner"])
    with ft[0]: ai_doctor_chat()
    with ft[1]: display_health_timeline()
    with ft[2]: medication_planner()


# =============================================================================
# MAIN ROUTER
# =============================================================================
def main():
    initialize_user_db()

    if st.session_state.username:
        app_main_content()
    else:
        # ── Branded sidebar (logged out) ──
        st.sidebar.markdown("""
        <div class="sidebar-brand">
            <span class="brand-icon">🏥</span>
            <span class="brand-name">GlucoCheck Pro+</span>
            <span class="brand-tag">Diabetes Management</span>
        </div>""", unsafe_allow_html=True)

        auth_option = st.sidebar.radio("mode", ["Login","Register"], label_visibility="collapsed")

        lc = "active" if auth_option == "Login"    else ""
        rc = "active" if auth_option == "Register" else ""
        st.sidebar.markdown(f"""
        <div class="auth-tab-row">
            <div class="auth-tab {lc}">🔑 Login</div>
            <div class="auth-tab {rc}">✨ Register</div>
        </div>""", unsafe_allow_html=True)

        if auth_option == "Register":
            st.sidebar.markdown('<span class="auth-section-label">Create your account</span>', unsafe_allow_html=True)
            nu  = st.sidebar.text_input("Username",          placeholder="Choose a username",  key="reg_u")
            np_ = st.sidebar.text_input("Password",          placeholder="Create a password",  type="password", key="reg_p")
            cp  = st.sidebar.text_input("Confirm Password",  placeholder="Repeat password",    type="password", key="reg_cp")
            if st.sidebar.button("✨ Create Account"):
                if not nu or not np_:         st.sidebar.error("Please fill all fields.")
                elif np_ != cp:               st.sidebar.error("Passwords don't match.")
                elif register_user(nu, np_):  st.sidebar.success("Account created! Please log in.")
                else:                         st.sidebar.error("Username already taken.")
        else:
            st.sidebar.markdown('<span class="auth-section-label">Sign in to your account</span>', unsafe_allow_html=True)
            uname = st.sidebar.text_input("Username", placeholder="Your username", key="login_u")
            pw    = st.sidebar.text_input("Password", placeholder="Your password", type="password", key="login_p")
            if st.sidebar.button("🔑 Login"):
                if verify_user(uname, pw):
                    st.session_state.token    = generate_token(uname)
                    st.session_state.username = uname
                    st.sidebar.success(f"Welcome, {uname}! 👋")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password.")

        st.sidebar.markdown("---")
        st.sidebar.markdown("""<div style="text-align:center;padding:4px 0;">
            <span style="font-size:0.72rem;color:rgba(255,255,255,0.28);">🔒 Your data is private &amp; secure</span>
        </div>""", unsafe_allow_html=True)

        # ── Landing page ──
        st.markdown("""
        <div class="hero-section fade-in">
            <div class="hero-badge">🏥 Advanced Diabetes Management</div>
            <h1 class="hero-title">GlucoCheck <span>Pro+</span></h1>
            <p class="hero-subtitle">
                AI-powered risk assessment, personalised health plans, and smart medication
                tracking — all in one platform built for people managing diabetes.
            </p>
            <div class="hero-features">
                <span class="hero-feature-pill">🎯 AI Risk Assessment</span>
                <span class="hero-feature-pill">🩺 Virtual Specialist</span>
                <span class="hero-feature-pill">💊 Medication Tracker</span>
                <span class="hero-feature-pill">📈 Health Timeline</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="stats-bar">
            <div class="stat-item"><span class="stat-number">95%</span><span class="stat-label">MODEL ACCURACY</span></div>
            <div class="stat-item"><span class="stat-number">5</span><span class="stat-label">RISK TIERS</span></div>
            <div class="stat-item"><span class="stat-number">AI</span><span class="stat-label">POWERED CHAT</span></div>
            <div class="stat-item"><span class="stat-number">PDF</span><span class="stat-label">HEALTH REPORTS</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header">
            <h2>Everything You Need</h2>
            <p>Comprehensive tools to monitor and manage your diabetes risk</p>
            <div class="section-divider"></div>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(4)
        for col, color, icon, title, desc in [
            (cols[0],"#6C63FF","🎯","Risk Assessment",  "Instant AI-powered diabetes risk score with personalised plans."),
            (cols[1],"#FF6B6B","🩺","AI Doctor Chat",   "Chat with our Gemini-powered virtual diabetes specialist."),
            (cols[2],"#4CAF50","💊","Medication Planner","Smart calendar tracking — never miss a dose again."),
            (cols[3],"#FFC107","📈","Health Timeline",  "Visualise risk trends and track your progress over time."),
        ]:
            with col:
                st.markdown(f"""
                <div class="feature-card" style="border-top-color:{color};">
                    <span class="feature-icon">{icon}</span>
                    <h3>{title}</h3><p>{desc}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <h2>How It Works</h2>
            <p>Three simple steps to better diabetes management</p>
            <div class="section-divider"></div>
        </div>""", unsafe_allow_html=True)

        sc = st.columns(3)
        for col, num, title, desc in [
            (sc[0],"01","Create Your Account","Register for free in seconds. Your data stays private."),
            (sc[1],"02","Enter Health Metrics","Input glucose, BMI, blood pressure and other vitals."),
            (sc[2],"03","Get Your Report",    "Personalised risk score + downloadable PDF report."),
        ]:
            with col:
                st.markdown(f"""
                <div class="step-card">
                    <span class="step-number">{num}</span>
                    <h4>{title}</h4><p>{desc}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 **Login or Register** using the sidebar to unlock all features.")


if __name__ == "__main__":
    main()

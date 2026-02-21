import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.card import card
from streamlit_extras.stylable_container import stylable_container
import time
import sys
import os
import google.generativeai as genai
import datetime
from streamlit_calendar import calendar
import pytz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import base64
import uuid
import shutil  # Added for file operations

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

# Deferred startup notices (to avoid using Streamlit APIs before page config)
startup_notices = []

# --- Clean up any existing confusion_matrix.html ---
confusion_matrix_path = os.path.join(current_dir, "confusion_matrix.html")
if os.path.exists(confusion_matrix_path):
    try:
        os.remove(confusion_matrix_path)
        startup_notices.append(("success", "Removed existing confusion_matrix.html file"))
    except Exception as e:
        startup_notices.append(("warning", f"Could not remove confusion_matrix.html: {e}"))

# --- Authentication Imports ---
try:
    from auth_utils import (
        initialize_user_db, hash_password, register_user, verify_user,
        generate_token, verify_token, update_password, delete_user
    )
except ImportError:
    startup_notices.append(("error", "Authentication utilities not found. Some features may be limited."))
    def initialize_user_db():
        pass
    def hash_password(p):
        return p
    def register_user(u, p):
        return False
    def verify_user(u, p):
        return False
    def generate_token(u):
        return "dummy_token"
    def verify_token(t):
        return None
    def update_password(u, p):
        return False
    def delete_user(u):
        return False

for level, message in startup_notices:
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


# --- Custom CSS ---
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #6C63FF;
        --secondary: #4D44DB;
        --danger: #FF6B6B;
        --success: #4CAF50;
        --warning: #FFC107;
        --dark: #1a1a2e;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* ── Sidebar dark theme ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%) !important;
        border-right: 1px solid rgba(108,99,255,0.25) !important;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.88) !important;
    }
    /* Sidebar inputs */
    [data-testid="stSidebar"] input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(108,99,255,0.4) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 10px 14px !important;
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color: rgba(255,255,255,0.35) !important;
    }
    /* Sidebar labels */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    /* Sidebar radio */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 6px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] span:last-child {
        color: rgba(255,255,255,0.82) !important;
    }
    /* Sidebar button */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #6C63FF, #4D44DB) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-top: 8px !important;
        box-shadow: 0 4px 16px rgba(108,99,255,0.4) !important;
        transition: all 0.3s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(108,99,255,0.55) !important;
    }
    /* Sidebar dividers and hr */
    [data-testid="stSidebar"] hr {
        border-color: rgba(108,99,255,0.25) !important;
        margin: 16px 0 !important;
    }
    /* Sidebar success/error messages */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    /* Hide default sidebar top padding */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem !important;
    }
    /* Sidebar markdown text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    /* Sidebar logo/branding card */
    .sidebar-brand {
        background: rgba(108,99,255,0.15);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 14px;
        padding: 18px 16px 14px;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-brand .brand-icon { font-size: 2.4rem; display: block; margin-bottom: 6px; }
    .sidebar-brand .brand-name {
        font-size: 1.2rem; font-weight: 700;
        color: white; display: block; margin-bottom: 2px;
    }
    .sidebar-brand .brand-tag {
        font-size: 0.72rem; color: rgba(255,255,255,0.5);
        letter-spacing: 1px; text-transform: uppercase;
    }
    /* Auth tab switcher */
    .auth-tab-row {
        display: flex;
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 20px;
        gap: 4px;
    }
    .auth-tab {
        flex: 1;
        text-align: center;
        padding: 8px 0;
        border-radius: 9px;
        font-size: 0.87rem;
        font-weight: 600;
        color: rgba(255,255,255,0.5);
        cursor: default;
    }
    .auth-tab.active {
        background: linear-gradient(135deg, #6C63FF, #4D44DB);
        color: white;
        box-shadow: 0 3px 10px rgba(108,99,255,0.4);
    }
    .auth-section-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(255,255,255,0.45) !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 4px;
        display: block;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #4D44DB) !important;
        color: white !important;
        border: none !important;
        padding: 10px 26px !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(108, 99, 255, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.5) !important;
    }

    /* ── Risk cards ── */
    .risk-card {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid var(--primary);
        background: white;
        transition: all 0.3s;
    }

    /* ── Hero section ── */
    .hero-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 64px 50px 56px;
        margin-bottom: 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -40%;
        left: -20%;
        width: 160%;
        height: 160%;
        background: radial-gradient(circle at 60% 40%, rgba(108,99,255,0.18) 0%, transparent 55%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(108,99,255,0.25);
        border: 1px solid rgba(108,99,255,0.5);
        color: #c4c0ff;
        padding: 5px 18px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 18px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 14px;
        line-height: 1.15;
        text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }
    .hero-title span {
        background: linear-gradient(90deg, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.7);
        margin-bottom: 36px;
        font-weight: 300;
        max-width: 560px;
        margin-left: auto;
        margin-right: auto;
    }
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 28px;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .hero-feature-pill {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.85);
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* ── Stats bar ── */
    .stats-bar {
        background: linear-gradient(135deg, #6C63FF 0%, #4D44DB 100%);
        border-radius: 16px;
        padding: 28px 20px;
        margin: 0 0 32px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
    }
    .stat-item { text-align: center; }
    .stat-number { font-size: 2rem; font-weight: 700; color: #fff; display: block; }
    .stat-label  { font-size: 0.78rem; color: rgba(255,255,255,0.7); letter-spacing: 0.5px; }

    /* ── Feature cards ── */
    .feature-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 28px 22px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
        border-top: 4px solid;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        text-align: center;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 14px 38px rgba(0,0,0,0.12);
    }
    .feature-icon { font-size: 2.4rem; margin-bottom: 12px; display: block; }
    .feature-card h3 { font-size: 1.05rem; font-weight: 600; color: #1a1a2e; margin-bottom: 8px; }
    .feature-card p  { font-size: 0.88rem; color: #666; line-height: 1.55; margin: 0; }

    /* ── How it works steps ── */
    .step-card {
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        border-radius: 14px;
        padding: 24px 18px;
        text-align: center;
        border: 1px solid rgba(108,99,255,0.12);
    }
    .step-number { font-size: 2.2rem; font-weight: 700; color: #6C63FF; display: block; margin-bottom: 6px; }
    .step-card h4 { font-size: 0.98rem; font-weight: 600; color: #1a1a2e; margin-bottom: 5px; }
    .step-card p  { font-size: 0.84rem; color: #555; margin: 0; }

    /* ── Section header ── */
    .section-header { text-align: center; margin: 38px 0 26px; }
    .section-header h2 { font-size: 1.75rem; font-weight: 700; color: #1a1a2e; margin-bottom: 6px; }
    .section-header p  { color: #666; font-size: 0.93rem; margin: 0; }
    .section-divider {
        width: 48px; height: 4px;
        background: linear-gradient(90deg, #6C63FF, #4D44DB);
        border-radius: 2px;
        margin: 8px auto 0;
    }

    /* ── Chat bubbles ── */
    .chat-container {
        background: #f8f7ff;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
        max-height: 420px;
        overflow-y: auto;
        border: 1px solid rgba(108,99,255,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #6C63FF, #4D44DB);
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 8px 0 8px auto;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(108,99,255,0.3);
        display: block;
    }
    .ai-message {
        background: white;
        color: #333;
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 8px auto 8px 0;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border: 1px solid rgba(0,0,0,0.06);
        display: block;
    }
    .message-timestamp { font-size: 0.68em; opacity: 0.65; margin-top: 4px; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.5s ease-out; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# --- AI Doctor Chat Class ---
class AIDoctorAssistant:
    def __init__(self):
        self.model = None
        self.init_error = None

        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.init_error = "Missing GEMINI_API_KEY in Streamlit secrets or environment."
            return

        try:
            genai.configure(api_key=api_key)

            model_options = [
                'models/gemini-1.5-flash',
                'models/gemini-1.5-flash-latest',
                'models/gemini-pro',
                'models/gemini-pro-vision'
            ]

            available_models = [
                m.name for m in genai.list_models()
                if 'generateContent' in m.supported_generation_methods
            ]

            for model_name in model_options:
                if model_name in available_models:
                    self.model = genai.GenerativeModel(model_name)
                    break

            if self.model is None:
                self.init_error = "No supported Gemini model available."

        except Exception as e:
            self.init_error = f"Failed to initialize Gemini: {e}"

        self.system_prompt = """You are a friendly and knowledgeable diabetes specialist AI assistant.
Provide accurate, evidence-based information and advise consulting a healthcare professional."""

    def generate_response(self, user_query, context=""):
        if not self.model:
            return "AI service unavailable. Please check configuration."
        
        try:
            response = self.model.generate_content(
                {
                    "parts": [
                        {"text": self.system_prompt},
                        {"text": f"Context: {context}"},
                        {"text": f"Question: {user_query}"}
                    ]
                },
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 200,
                    "top_p": 0.9
                }
            )
            return response.text
        except Exception:
            return "I'm having trouble responding right now. Please try again later."

# --- Session State Initialization ---
def init_session_state():
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'medications' not in st.session_state:
        st.session_state.medications = []
    if 'health_history' not in st.session_state:
        st.session_state.health_history = []
    if 'last_assessment' not in st.session_state:
        st.session_state.last_assessment = None
    # ✅ FIX: Added session state for chat input text
    if 'chat_input_text' not in st.session_state:
        st.session_state.chat_input_text = ""

init_session_state()

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = os.path.join(current_dir, "model.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")

    confusion_matrix_path = os.path.join(current_dir, "confusion_matrix.html")
    if os.path.exists(confusion_matrix_path):
        try:
            os.remove(confusion_matrix_path)
        except Exception as e:
            st.warning(f"Could not remove confusion_matrix.html during model load: {e}")

    model = None
    scaler = None
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.warning(f"Model file not found: {model_path}")
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            st.warning(f"Scaler file not found: {scaler_path}")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
    return model, scaler

model, scaler = load_model()

# --- Core Functions ---
def get_personalized_recommendations(probability, bmi, age):
    """5-tier recommendation system with personalized plans"""
    risk_level = ""
    color = ""
    icon = ""
    diet = []
    exercise = []
    supplements = []
    bmi_category = ""
    age_group = ""

    if age < 18:
        age_group = "child/teen"
    elif 18 <= age <= 60:
        age_group = "adult"
    else:
        age_group = "senior"

    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal Weight"
    elif 25.0 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    if probability < 0.2:
        risk_level = "Very Low"
        color = "#4CAF50"
        icon = "😊"
        diet = ["Maintain balanced diet with whole foods.", "Limit processed sugars and unhealthy fats.", "Stay hydrated."]
        exercise = ["Regular moderate activity (e.g., brisk walking 30 min, 5 times/week).", "Incorporate strength training 2-3 times/week."]
        supplements = ["Multivitamin (if diet lacks variety).", "Omega-3 fatty acids."]
    elif 0.2 <= probability < 0.4:
        risk_level = "Low"
        color = "#8BC34A"
        icon = "🙂"
        diet = ["Focus on portion control.", "Increase fiber intake (fruits, vegetables, whole grains).", "Reduce sugary drinks."]
        exercise = ["Aim for 150 minutes of moderate-intensity exercise weekly.", "Try activities like cycling, swimming, or dancing."]
        supplements = ["Vitamin D (especially if limited sun exposure)."]
    elif 0.4 <= probability < 0.6:
        risk_level = "Moderate"
        color = "#FFC107"
        icon = "😐"
        diet = ["Strictly limit refined carbs and added sugars.", "Emphasize lean proteins and healthy fats.", "Consider consulting a nutritionist."]
        exercise = ["Increase physical activity to 200-250 minutes of moderate intensity per week.", "Include both cardio and strength exercises."]
        supplements = ["Chromium picolinate (may help with insulin sensitivity).", "Magnesium."]
    elif 0.6 <= probability < 0.8:
        risk_level = "High"
        color = "#FF9800"
        icon = "😟"
        diet = ["Adopt a low-glycemic index diet.", "Eliminate sugary snacks and drinks.", "Work with a dietitian for a personalized meal plan."]
        exercise = ["Intensify exercise to 250-300 minutes of moderate activity, or 150 minutes of vigorous activity.", "Monitor blood sugar before/after exercise."]
        supplements = ["Berberine (natural compound with glucose-lowering effects, consult doctor).", "Alpha-lipoic acid."]
    else:
        risk_level = "Very High"
        color = "#F44336"
        icon = "❗"
        diet = ["Follow a medical nutrition therapy plan.", "Avoid all processed foods and sugary items.", "Regular blood sugar monitoring is crucial."]
        exercise = ["Daily physical activity, even short walks, is beneficial.", "Supervised exercise program recommended.", "Consult doctor before starting new regimens."]
        supplements = ["Only take supplements under strict medical supervision.", "Coenzyme Q10 (for general heart health)."]

    return {
        "risk_level": risk_level,
        "color": color,
        "icon": icon,
        "diet": diet,
        "exercise": exercise,
        "supplements": supplements,
        "bmi_category": bmi_category,
        "age_group": age_group
    }

def create_gauge(probability, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Score (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#4CAF50"},
                {'range': [20, 40], 'color': "#8BC34A"},
                {'range': [40, 60], 'color': "#FFC107"},
                {'range': [60, 80], 'color': "#FF9800"},
                {'range': [80, 100], 'color': "#F44336"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        }))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def risk_card(title, value, color, icon):
    st.markdown(f"""
    <div class="risk-card" style="border-left: 5px solid {color};">
        <h3 style="color: {color}; margin-top: 0; margin-bottom: 10px;">{icon} {title}</h3>
        <p style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

def generate_pdf_report(probability, recommendations, age, bmi, glucose, bp, skin_thickness, insulin, pedigree, pregnancies):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 24)
    c.setFillColor("#6C63FF")
    c.drawString(50, height - 50, "GlucoCheck Pro+ Health Report")

    c.setFont("Helvetica", 12)
    c.setFillColor("gray")
    c.drawString(50, height - 70, f"Date: {datetime.date.today().strftime('%Y-%m-%d')}")

    y_pos = height - 120
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor("black")
    c.drawString(50, y_pos, "Patient Information:")
    c.setFont("Helvetica", 12)
    c.drawString(70, y_pos - 20, f"Age: {age} years")
    c.drawString(70, y_pos - 40, f"BMI: {bmi:.1f} ({recommendations['bmi_category']})")
    c.drawString(70, y_pos - 60, f"Glucose: {glucose} mg/dL")
    c.drawString(70, y_pos - 80, f"Blood Pressure: {bp} mmHg")
    c.drawString(70, y_pos - 100, f"Skin Thickness: {skin_thickness} mm")
    c.drawString(70, y_pos - 120, f"Insulin: {insulin} μU/mL")
    c.drawString(70, y_pos - 140, f"Diabetes Pedigree Function: {pedigree:.3f}")
    c.drawString(70, y_pos - 160, f"Pregnancies: {pregnancies}")

    y_pos = y_pos - 200
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Diabetes Risk Assessment:")
    c.setFont("Helvetica", 12)
    c.drawString(70, y_pos - 20, f"Calculated Probability: {probability*100:.1f}%")
    c.setFillColor(recommendations['color'])
    c.drawString(70, y_pos - 40, f"Overall Risk Level: {recommendations['risk_level']}")
    c.setFillColor("black")

    y_pos = y_pos - 80
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Personalized Recommendations:")

    y_pos -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, y_pos, "Diet Plan:")
    c.setFont("Helvetica", 12)
    for item in recommendations["diet"]:
        y_pos -= 15
        c.drawString(80, y_pos, f"- {item}")

    y_pos -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, y_pos, "Exercise Plan:")
    c.setFont("Helvetica", 12)
    for item in recommendations["exercise"]:
        y_pos -= 15
        c.drawString(80, y_pos, f"- {item}")
    
    y_pos -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(70, y_pos, "Suggested Supplements:")
    c.setFont("Helvetica", 12)
    for item in recommendations["supplements"]:
        y_pos -= 15
        c.drawString(80, y_pos, f"- {item}")
    
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor("gray")
    c.drawString(50, 50, "Disclaimer: This report provides a risk assessment and general recommendations.")
    c.drawString(50, 35, "Always consult a qualified healthcare professional for diagnosis and treatment.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ✅ MAIN FIX: Replaced st.chat_input() with st.text_input() + button
# st.chat_input() is NOT allowed inside st.tabs(), st.columns(), st.expander(),
# st.form(), or st.sidebar. Using text_input + button works everywhere.
def ai_doctor_chat():
    st.header("🩺 AI Diabetes Specialist")
    st.caption("Get answers to your diabetes-related questions (not a substitute for medical advice)")
    
    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = AIDoctorAssistant()

    assistant = st.session_state.ai_assistant

    if assistant.init_error:
        st.warning(f"AI Doctor unavailable: {assistant.init_error}")
    
    # Chat history display
    chat_html = "<div class='chat-container'>"
    for message in st.session_state.chat_history:
        display_timestamp = datetime.datetime.strptime(
            message['timestamp'], "%Y%m%d%H%M%S%f"
        ).strftime("%H:%M")
        
        if message['role'] == 'user':
            chat_html += f"""
            <div class='user-message'>
                {message['content']}
                <div class='message-timestamp' style='text-align: right;'>{display_timestamp}</div>
            </div>
            """
        else:
            chat_html += f"""
            <div class='ai-message'>
                <strong>AI Doctor:</strong> {message['content']}
                <div class='message-timestamp'>{display_timestamp}</div>
            </div>
            """
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # ✅ FIX: Use text_input + button instead of st.chat_input()
    # st.chat_input() raises StreamlitAPIException inside st.tabs()
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Your question",
            placeholder="Ask your diabetes-related question...",
            label_visibility="collapsed",
            key="chat_text_input"
        )
    with col_btn:
        send_clicked = st.button("Send 💬", use_container_width=True)

    if send_clicked and user_query and user_query.strip():
        granular_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query.strip(),
            "timestamp": granular_timestamp
        })
        
        context = ""
        if st.session_state.health_history:
            latest = st.session_state.health_history[-1]
            context = (
                f"User's latest health stats: {latest['risk_level']} risk, "
                f"BMI {latest['bmi']}, Glucose {latest['glucose']}, Age {latest['age']}"
            )
        
        with st.spinner("AI Doctor is thinking..."):
            ai_response = assistant.generate_response(user_query.strip(), context)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        })
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# --- Health Timeline ---
def display_health_timeline():
    st.header("📈 Your Health Timeline")
    
    if not st.session_state.health_history:
        st.info("No health assessments recorded yet. Complete an assessment to see your timeline.")
        return
    
    timeline_data = pd.DataFrame(st.session_state.health_history)
    timeline_data['date'] = pd.to_datetime(timeline_data['date'])
    timeline_data.sort_values('date', inplace=True)
    
    fig = px.scatter(
        timeline_data,
        x='date',
        y='probability',
        color='risk_level',
        size='probability',
        hover_data=['bmi', 'glucose', 'age'],
        title="Your Diabetes Risk Over Time",
        color_discrete_map={
            "Very Low": "#4CAF50",
            "Low": "#8BC34A",
            "Moderate": "#FFC107",
            "High": "#FF9800",
            "Very High": "#F44336"
        }
    )
    
    fig.update_layout(
        yaxis_title="Diabetes Probability (%)",
        xaxis_title="Date",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📝 View Detailed History"):
        for idx, entry in enumerate(reversed(st.session_state.health_history)):
            unique_key = f"entry_{entry['date']}_{str(uuid.uuid4())[:8]}"
            
            with stylable_container(
                key=unique_key,
                css_styles="""
                {
                    border: 1px solid rgba(0,0,0,0.1);
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                """
            ):
                cols = st.columns([1,3])
                with cols[0]:
                    st.markdown(f"**{entry['date']}**")
                    st.markdown(f"**Risk**: {entry['risk_level']}")
                with cols[1]:
                    st.markdown(f"**Probability**: {entry['probability']*100:.1f}%")
                    st.markdown(f"**BMI**: {entry['bmi']:.1f} | **Glucose**: {entry['glucose']} mg/dL")
                    st.markdown(f"**Age**: {entry['age']}")


# --- Medication Planner ---
def medication_planner():
    st.header("💊 Medication & Reminder Planner")
    
    with st.form("add_medication"):
        st.subheader("Add New Medication")
        cols = st.columns(2)
        with cols[0]:
            name = st.text_input("Medication Name", placeholder="Metformin")
        with cols[1]:
            dosage = st.text_input("Dosage", placeholder="500mg")
        
        cols = st.columns(2)
        with cols[0]:
            frequency = st.selectbox("Frequency", ["Once daily", "Twice daily", "Three times daily", "Weekly"])
        with cols[1]:
            time_of_day = st.multiselect("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        
        start_date = st.date_input("Start Date", datetime.date.today())
        notes = st.text_area("Additional Notes")
        
        if st.form_submit_button("➕ Add Medication"):
            if name and dosage and time_of_day:
                new_med = {
                    "name": name,
                    "dosage": dosage,
                    "frequency": frequency,
                    "times": time_of_day,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "notes": notes,
                    "reminders": True
                }
                st.session_state.medications.append(new_med)
                st.success("Medication added successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields")
    
    st.subheader("Your Medications")
    if not st.session_state.medications:
        st.info("No medications added yet.")
    else:
        for i, med in enumerate(st.session_state.medications):
            unique_key = f"med_{i}_{med['name']}_{str(uuid.uuid4())[:8]}"
            
            with stylable_container(
                key=unique_key,
                css_styles="""
                {
                    border-left: 4px solid #6C63FF;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 8px;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                """
            ):
                cols = st.columns([3,1])
                with cols[0]:
                    st.markdown(f"""
                    **{med['name']}** ({med['dosage']})
                    - **Frequency**: {med['frequency']}
                    - **Times**: {', '.join(med['times'])}
                    - **Start Date**: {med['start_date']}
                    """)
                    if med.get('notes'):
                        st.markdown(f"**Notes**: {med['notes']}")
                with cols[1]:
                    if st.button("❌ Remove", key=f"remove_{i}_{med['name']}"):
                        st.session_state.medications.pop(i)
                        st.rerun()
    
    st.subheader("📅 Medication Schedule")
    if st.session_state.medications:
        events = []
        colors = ["#6C63FF", "#4D44DB", "#FF6B6B", "#4CAF50", "#FFC107"]
        
        for i, med in enumerate(st.session_state.medications):
            for time_period in med['times']:
                start_time_str = get_time_for_period(time_period, end=False)
                end_time_str = get_time_for_period(time_period, end=True)
                for d_offset in range(7):
                    event_date = (datetime.date.today() + datetime.timedelta(days=d_offset)).isoformat()
                    events.append({
                        "title": f"{med['name']} ({med['dosage']})",
                        "color": colors[i % len(colors)],
                        "start": f"{event_date}T{start_time_str}",
                        "end": f"{event_date}T{end_time_str}"
                    })
        
        calendar_options = {
            "headerToolbar": {
                "left": "today prev,next",
                "center": "title",
                "right": "dayGridMonth,timeGridWeek,timeGridDay"
            },
            "initialView": "timeGridWeek",
            "slotMinTime": "06:00:00",
            "slotMaxTime": "22:00:00",
            "navLinks": True,
            "selectable": True,
            "editable": False,
            "dayMaxEvents": True,
            "height": "auto",
            "aspectRatio": 2
        }
        
        calendar(events=events, options=calendar_options, key="medication_calendar")
    else:
        st.info("Add medications to see your schedule")

def get_time_for_period(period, end=False):
    times = {
        "Morning": ("08:00:00", "09:00:00"),
        "Afternoon": ("12:00:00", "13:00:00"),
        "Evening": ("18:00:00", "19:00:00"),
        "Night": ("21:00:00", "22:00:00")
    }
    return times[period][1] if end else times[period][0]


# --- Main App Content ---
def app_main_content():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://supergut-production.s3.amazonaws.com/uploads/finger_prick_animated_6c444158e0.jpg", width=120)
    with col2:
        st.title("GlucoCheck Pro+")
        st.caption("Advanced Diabetes Management with AI Assistance")

    st.markdown("---")

    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
        
        st.title("Health Assessment")
        with st.form("input_form"):
            st.subheader("Health Metrics")
            age = st.slider("Age", 1, 100, 30)
            pregnancies = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.slider("Glucose (mg/dL)", 50, 300, 100)
            bp = st.slider("Blood Pressure (mmHg)", 40, 180, 70)
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
            insulin = st.slider("Insulin (μU/mL)", 0, 1000, 80)
            bmi = st.slider("BMI", 10.0, 70.0, 25.0)
            pedigree = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5)

            submitted = st.form_submit_button("Assess My Risk", type="primary")

    if submitted:
        if model is None or scaler is None:
            st.error("Cannot perform risk assessment: Model or scaler not loaded.")
            return

        input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, pedigree, age]])
        input_scaled = scaler.transform(input_data)

        probability = model.predict_proba(input_scaled)[0][1]
        recommendations = get_personalized_recommendations(probability, bmi, age)

        new_entry = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "probability": float(probability),
            "risk_level": recommendations["risk_level"],
            "bmi": float(bmi),
            "glucose": float(glucose),
            "age": int(age),
            "bp": float(bp)
        }
        st.session_state.health_history.append(new_entry)
        st.session_state.last_assessment = new_entry

        with st.container():
            st.success("Analysis complete! Here's your personalized health report:")
            
            pdf_bytes = generate_pdf_report(probability, recommendations, age, bmi, glucose, bp, skin_thickness, insulin, pedigree, pregnancies)
            st.download_button(
                label="📥 Download Full Report (PDF)",
                data=pdf_bytes,
                file_name=f"diabetes_report_{datetime.date.today()}.pdf",
                mime="application/pdf"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                risk_card("Risk Level", recommendations["risk_level"], recommendations["color"], recommendations["icon"])
            with col2:
                risk_card("Probability", f"{probability*100:.1f}%", recommendations["color"], "📊")
            with col3:
                risk_card("BMI Category", recommendations["bmi_category"], "#6C63FF", "⚖️")

            st.markdown("---")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.plotly_chart(create_gauge(probability, recommendations["color"]), use_container_width=True)

                with st.expander("📊 Understanding Your Risk", expanded=True):
                    st.markdown(f"""
                    - Your risk level: **{recommendations['risk_level']}**
                    - Age group: **{recommendations['age_group'].title()}**
                    - BMI classification: **{recommendations['bmi_category']}**
                    """)

                    if probability > 0.6:
                        st.warning("Consider consulting an endocrinologist for further evaluation")
                    elif probability > 0.4:
                        st.info("Lifestyle modifications can significantly reduce your risk")
                    else:
                        st.success("Maintain your healthy habits!")

            with col2:
                tab1, tab2, tab3 = st.tabs(["🍽️ Diet Plan", "🏋️ Exercise Plan", "💊 Supplements"])

                with tab1:
                    st.subheader("Personalized Nutrition Guide")
                    for item in recommendations["diet"]:
                        st.markdown(f"- {item}")

                    st.markdown("### Sample Meal Plan")
                    if recommendations["risk_level"] in ["High", "Very High"]:
                        st.markdown("""
                        - **Breakfast**: Veggie omelet + avocado
                        - **Snack**: Handful of almonds
                        - **Lunch**: Grilled salmon + quinoa + broccoli
                        - **Snack**: Greek yogurt + berries
                        - **Dinner**: Chicken stir-fry with mixed vegetables
                        """)
                    else:
                        st.markdown("""
                        - **Breakfast**: Oatmeal + nuts + berries
                        - **Snack**: Apple + peanut butter
                        - **Lunch**: Whole grain wrap with lean protein
                        - **Snack**: Cottage cheese + cucumber
                        - **Dinner**: Baked fish + sweet potato + greens
                        """)

                with tab2:
                    st.subheader("Fitness Recommendations")
                    for item in recommendations["exercise"]:
                        st.markdown(f"- {item}")

                    st.markdown("### Weekly Plan")
                    cols = st.columns(7)
                    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    for i, day in enumerate(days):
                        with cols[i]:
                            st.caption(day)
                            if recommendations["risk_level"] in ["High", "Very High"]:
                                st.write("🏋️‍♂️" if i % 2 else "🏃‍♂️")
                            else:
                                st.write("🧘‍♀️" if i % 3 == 0 else "🚶‍♂️")

                with tab3:
                    st.subheader("Suggested Supplements")
                    for item in recommendations["supplements"]:
                        st.markdown(f"- {item}")

                    st.markdown("### Supplement Guide")
                    st.write("Always consult your doctor before starting new supplements")

        st.markdown("---")
        st.header("Health Insights")

        tab1, tab2 = st.tabs(["Risk Factors", "Glucose Analysis"])

        with tab1:
            if model is not None and hasattr(model, 'feature_importances_'):
                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                
                feat_imp = pd.DataFrame({
                    'Factor': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig = px.bar(feat_imp, x='Importance', y='Factor', orientation='h',
                             title='What Impacts Your Diabetes Risk Most?',
                             color='Importance',
                             color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance data not available.")

        with tab2:
            try:
                df = pd.read_csv(os.path.join(current_dir, "enhanced_diabetes.csv"))
                fig = px.scatter(df, x="Glucose", y="BMI", color="Outcome",
                                 hover_data=["Age"],
                                 title="Glucose vs BMI Relationship",
                                 color_discrete_map={0: "#4CAF50", 1: "#F44336"},
                                 trendline="lowess")
                st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError:
                st.warning("Sample dataset 'enhanced_diabetes.csv' not found for visualization.")
            except Exception as e:
                st.error(f"An error occurred loading or plotting 'enhanced_diabetes.csv': {e}")

    else:
        # Landing Page when no submission
        with st.container():
            st.header("Welcome to GlucoCheck Pro+")
            st.markdown("""
            **Your comprehensive diabetes management platform** with:
            - 🎯 AI-powered risk assessment
            - 🍏 Personalized health plans
            - 🏋️‍♂️ Medication tracking & reminders
            - 🩺 Virtual diabetes specialist
            """)

            cols = st.columns(3)
            with cols[0]:
                card(
                    title="AI Health Assistant",
                    text="Get answers to your diabetes questions",
                    image="https://engineering.tamu.edu/news/2018/_news-images/isen-news-diabetes-image-4june2018.jpg"
                )
            with cols[1]:
                card(
                    title="Medication Tracker",
                    text="Never miss a dose with smart reminders",
                    image="https://domf5oio6qrcr.cloudfront.net/medialibrary/11693/1a3d7208-475c-4cce-8787-d2ca822f28e7.jpg"
                )
            with cols[2]:
                card(
                    title="Health Timeline",
                    text="Track your progress over time",
                    image="https://vidhilegalpolicy.in/wp-content/uploads/2021/10/mental-health-wellness-during-covid-19.jpg"
                )

            st.markdown("---")
            st.subheader("How It Works")
            steps = st.columns(3)
            with steps[0]:
                st.markdown("""
                ### 1️⃣ Enter Your Metrics
                Provide health information in the sidebar
                """)
            with steps[1]:
                st.markdown("""
                ### 2️⃣ Get Your Analysis
                Receive personalized risk assessment
                """)
            with steps[2]:
                st.markdown("""
                ### 3️⃣ Use Premium Features
                Chat with AI, track meds, view history
                """)

    # --- Premium Features Section ---
    st.markdown("---")
    st.header("✨ Premium Features")
    
    # ✅ FIX: All three features remain in tabs. The chat now uses text_input+button
    # which is fully compatible with st.tabs() unlike st.chat_input()
    feature_tabs = st.tabs(["🩺 AI Doctor Chat", "📈 Health Timeline", "💊 Medication Planner"])
    
    with feature_tabs[0]:
        ai_doctor_chat()
    
    with feature_tabs[1]:
        display_health_timeline()
    
    with feature_tabs[2]:
        medication_planner()


# --- Main App Router ---
def main():
    initialize_user_db()

    if st.session_state.username:
        app_main_content()
    else:
        # ── Branded Sidebar ──
        st.sidebar.markdown("""
        <div class="sidebar-brand">
            <span class="brand-icon">🏥</span>
            <span class="brand-name">GlucoCheck Pro+</span>
            <span class="brand-tag">Diabetes Management</span>
        </div>
        """, unsafe_allow_html=True)

        auth_option = st.sidebar.radio(
            "Choose an option",
            ["Login", "Register"],
            horizontal=False,
            label_visibility="collapsed"
        )

        # Show active tab indicator
        login_active  = "active" if auth_option == "Login"    else ""
        reg_active    = "active" if auth_option == "Register" else ""
        st.sidebar.markdown(f"""
        <div class="auth-tab-row">
            <div class="auth-tab {login_active}">🔑 Login</div>
            <div class="auth-tab {reg_active}">✨ Register</div>
        </div>
        """, unsafe_allow_html=True)

        if auth_option == "Register":
            st.sidebar.markdown('<span class="auth-section-label">Create your account</span>', unsafe_allow_html=True)
            new_username     = st.sidebar.text_input("Username",         placeholder="Choose a username",  key="reg_username")
            new_password     = st.sidebar.text_input("Password",         placeholder="Create a password",  type="password", key="reg_password")
            confirm_password = st.sidebar.text_input("Confirm Password", placeholder="Repeat password",    type="password", key="reg_confirm_password")

            if st.sidebar.button("✨ Create Account"):
                if not new_username or not new_password:
                    st.sidebar.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.sidebar.error("Passwords don't match")
                else:
                    if register_user(new_username, new_password):
                        st.sidebar.success("Account created! Please log in.")
                    else:
                        st.sidebar.error("Username already taken")
        else:
            st.sidebar.markdown('<span class="auth-section-label">Sign in to your account</span>', unsafe_allow_html=True)
            username = st.sidebar.text_input("Username", placeholder="Enter your username", key="login_username")
            password = st.sidebar.text_input("Password", placeholder="Enter your password", type="password", key="login_password")

            if st.sidebar.button("🔑 Login"):
                if verify_user(username, password):
                    st.session_state.token    = generate_token(username)
                    st.session_state.username = username
                    st.sidebar.success(f"Welcome back, {username}! 👋")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")

        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="text-align:center; padding: 8px 0;">
            <span style="font-size:0.75rem; color:rgba(255,255,255,0.35); letter-spacing:0.5px;">
                🔒 Your data is private & secure
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Beautiful Landing Page for non-authenticated users ──
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
        </div>
        """, unsafe_allow_html=True)

        # Stats bar
        st.markdown("""
        <div class="stats-bar">
            <div class="stat-item">
                <span class="stat-number">95%</span>
                <span class="stat-label">MODEL ACCURACY</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">5</span>
                <span class="stat-label">RISK TIERS</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">AI</span>
                <span class="stat-label">POWERED CHAT</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">PDF</span>
                <span class="stat-label">HEALTH REPORTS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("""
        <div class="section-header">
            <h2>Everything You Need</h2>
            <p>Comprehensive tools to monitor and manage your diabetes risk</p>
            <div class="section-divider"></div>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns(4)
        features = [
            ("#6C63FF", "🎯", "Risk Assessment", "Enter your health metrics and get an instant AI-powered diabetes risk score with personalised recommendations."),
            ("#FF6B6B", "🩺", "AI Doctor Chat", "Chat with our Gemini-powered virtual diabetes specialist and get evidence-based answers to your questions."),
            ("#4CAF50", "💊", "Medication Planner", "Track all your medications with a smart calendar view and never miss a dose again."),
            ("#FFC107", "📈", "Health Timeline", "Visualise your risk scores over time and see the impact of your lifestyle changes."),
        ]
        for col, (color, icon, title, desc) in zip([fc1, fc2, fc3, fc4], features):
            with col:
                st.markdown(f"""
                <div class="feature-card" style="border-top-color: {color};">
                    <span class="feature-icon">{icon}</span>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # How it works
        st.markdown("""
        <div class="section-header">
            <h2>How It Works</h2>
            <p>Three simple steps to better diabetes management</p>
            <div class="section-divider"></div>
        </div>
        """, unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        for col, num, title, desc in [
            (s1, "1", "Create Your Account", "Register for free in seconds using the sidebar. Your data stays private and secure."),
            (s2, "2", "Enter Health Metrics", "Input your glucose, BMI, blood pressure and other vitals into the sidebar form."),
            (s3, "3", "Get Your Report", "Receive a personalised risk score, recommendations, and a downloadable PDF report."),
        ]:
            with col:
                st.markdown(f"""
                <div class="step-card">
                    <span class="step-number">0{num}</span>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👈 **Login or Register** using the sidebar to unlock all features.")

if __name__ == "__main__":
    main()

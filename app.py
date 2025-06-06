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
import uuid  # Added for unique key generation

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
auth_dir = os.path.join(current_dir, 'auth')
if auth_dir not in sys.path:
    sys.path.append(auth_dir)

# --- Authentication Imports ---
try:
    from auth_utils import (
        initialize_user_db, hash_password, register_user, verify_user,
        generate_token, verify_token, update_password, delete_user
    )
except ImportError:
    st.error("Authentication utilities not found. Some features may be limited.")
    # Dummy functions to prevent crashes
    def initialize_user_db(): pass
    def hash_password(p): return p
    def register_user(u, p): return False
    def verify_user(u, p): return False
    def generate_token(u): return "dummy_token"
    def verify_token(t): return None
    def update_password(u, p): return False
    def delete_user(u): return False

# --- Page Config ---
st.set_page_config(
    page_title="GlucoCheck Pro+",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #6C63FF;
        --secondary: #4D44DB;
        --danger: #FF6B6B;
        --success: #4CAF50;
        --warning: #FFC107;
    }
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton>button {
        background: var(--primary);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(108, 99, 255, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: white;
        box-shadow: 5px 0 15px rgba(0,0,0,0.05);
    }
    
    .risk-card {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid var(--primary);
        transition: all 0.3s;
    }
    
    .chat-container {
        background-color: transparent;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    .user-message {
        background-color: #6C63FF;
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 8px 0 8px auto;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .ai-message {
        background-color: #f0f2f6;
        color: #333;
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 8px auto 8px 0;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .message-timestamp {
        font-size: 0.7em;
        opacity: 0.8;
        margin-top: 4px;
    }
    
    .medication-card {
        border-left: 4px solid #6C63FF;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# --- AI Doctor Chat Class ---
class AIDoctorAssistant:
    def __init__(self):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            
            model_options = [
                'models/gemini-1.5-flash',
                'models/gemini-1.5-flash-latest',
                'models/gemini-pro',
                'models/gemini-pro-vision'
            ]
            
            available_models = [m.name for m in genai.list_models() 
                              if 'generateContent' in m.supported_generation_methods]
            
            for model_name in model_options:
                if model_name in available_models:
                    self.model = genai.GenerativeModel(model_name)
                    st.success(f"Using model: {model_name}")
                    break
            else:
                raise ValueError("No supported Gemini model available")
                
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}")
            self.model = None
        
        self.system_prompt = """You are a friendly and knowledgeable diabetes specialist AI assistant. 
        Provide accurate, evidence-based information about diabetes prevention, management, and treatment.
        Be empathetic and supportive in your responses. If a question is outside your scope or requires 
        medical attention, advise consulting a healthcare professional."""

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
        except Exception as e:
            st.error(f"Gemini Error: {str(e)}")
            return "I'm having trouble responding. Please try again later."

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

init_session_state()

# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = os.path.join(current_dir, "model.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")

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

    # Determine age group
    if age < 18:
        age_group = "child/teen"
    elif 18 <= age <= 60:
        age_group = "adult"
    else:
        age_group = "senior"

    # Determine BMI category
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
    """Create risk gauge visualization"""
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
    """Create styled risk card"""
    st.markdown(f"""
    <div class="risk-card" style="border-left: 5px solid {color};">
        <h3 style="color: {color}; margin-top: 0; margin-bottom: 10px;">{icon} {title}</h3>
        <p style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

def generate_pdf_report(probability, recommendations, age, bmi, glucose, bp, skin_thickness, insulin, pedigree, pregnancies):
    """Generate PDF health report"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor("#6C63FF")
    c.drawString(50, height - 50, "GlucoCheck Pro+ Health Report")

    # Date
    c.setFont("Helvetica", 12)
    c.setFillColor("gray")
    c.drawString(50, height - 70, f"Date: {datetime.date.today().strftime('%Y-%m-%d')}")

    # Patient Information
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

    # Risk Assessment
    y_pos = y_pos - 200
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, "Diabetes Risk Assessment:")
    c.setFont("Helvetica", 12)
    c.drawString(70, y_pos - 20, f"Calculated Probability: {probability*100:.1f}%")
    c.setFillColor(recommendations['color'])
    c.drawString(70, y_pos - 40, f"Overall Risk Level: {recommendations['risk_level']}")
    c.setFillColor("black")

    # Recommendations
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
    
    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor("gray")
    c.drawString(50, 50, "Disclaimer: This report provides a risk assessment and general recommendations.")
    c.drawString(50, 35, "Always consult a qualified healthcare professional for diagnosis and treatment.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# --- AI Doctor Chat ---
def ai_doctor_chat():
    st.header("🩺 AI Diabetes Specialist")
    st.caption("Get answers to your diabetes-related questions (not a substitute for medical advice)")
    
    assistant = AIDoctorAssistant()
    
    # Chat history display
    chat_html = "<div class='chat-container'>"
    for message in st.session_state.chat_history:
        display_timestamp = datetime.datetime.strptime(message['timestamp'], "%Y%m%d%H%M%S%f").strftime("%H:%M")
        
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
    
    # Chat input
    user_query = st.chat_input("Ask your diabetes-related question...")
    if user_query:
        granular_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": granular_timestamp
        })
        
        context = ""
        if st.session_state.health_history:
            latest = st.session_state.health_history[-1]
            context = f"User's latest health stats: {latest['risk_level']} risk, BMI {latest['bmi']}, Glucose {latest['glucose']}, Age {latest['age']}"
        
        with st.spinner("AI Doctor is thinking..."):
            ai_response = assistant.generate_response(user_query, context)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        })
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
    
    # Detailed history with unique keys
    with st.expander("📝 View Detailed History"):
        for idx, entry in enumerate(reversed(st.session_state.health_history)):
            # Generate unique key using UUID
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
    
    # Add new medication
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
    
    # Current medications list with unique keys
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
    
    # Calendar view
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
        
        calendar_component = calendar(events=events, options=calendar_options, key="medication_calendar")
    else:
        st.info("Add medications to see your schedule")

def get_time_for_period(period, end=False):
    """Helper function for medication planner"""
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
        st.image("https://i.imgur.com/JqYeZvn.png", width=120)
    with col2:
        st.title("GlucoCheck Pro+")
        st.caption("Advanced Diabetes Management with AI Assistance")

    st.markdown("---")

    # Sidebar Input
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
                file_name=f"glucocheck_report_{datetime.date.today()}.pdf",
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
                # Ensure 'enhanced_diabetes.csv' is available for this part to work.
                # It should be in the same directory as the main script.
                df = pd.read_csv(os.path.join(current_dir, "enhanced_diabetes.csv")) 
                fig = px.scatter(df, x="Glucose", y="BMI", color="Outcome",
                                 hover_data=["Age"],
                                 title="Glucose vs BMI Relationship",
                                 color_discrete_map={0: "#4CAF50", 1: "#F44336"},
                                 trendline="lowess")
                st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError:
                st.warning("Sample dataset 'enhanced_diabetes.csv' not found for visualization. Please ensure it's in the correct directory.")
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
                    image="https://i.imgur.com/JqYeZvn.png"
                )
            with cols[1]:
                card(
                    title="Medication Tracker",
                    text="Never miss a dose with smart reminders",
                    image="https://i.imgur.com/vm6Wg8h.png"
                )
            with cols[2]:
                card(
                    title="Health Timeline",
                    text="Track your progress over time",
                    image="https://i.imgur.com/5XQZQ9u.png"
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

    # --- New Premium Features Section ---
    st.markdown("---")
    st.header("✨ Premium Features")
    
    feature_tabs = st.tabs(["AI Doctor Chat", "Health Timeline", "Medication Planner"])
    
    with feature_tabs[0]:
        ai_doctor_chat()
    
    with feature_tabs[1]:
        display_health_timeline()
    
    with feature_tabs[2]:
        medication_planner()

# --- Main App Router ---
def main():
    initialize_user_db() # Ensure user database is ready

    if st.session_state.username:
        app_main_content()
    else:
        # Authentication UI
        st.sidebar.header("Authentication")
        auth_option = st.sidebar.radio("Choose an option", ["Login", "Register"])

        if auth_option == "Register":
            st.sidebar.subheader("New User Registration")
            new_username = st.sidebar.text_input("New Username", key="reg_username")
            new_password = st.sidebar.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="reg_confirm_password")

            if st.sidebar.button("Register Account"):
                if new_password == confirm_password:
                    if register_user(new_username, new_password):
                        st.sidebar.success("Registration successful! Please log in.")
                    else:
                        st.sidebar.error("Username already exists")
                else:
                    st.sidebar.error("Passwords don't match")
        else:
            st.sidebar.subheader("User Login")
            username = st.sidebar.text_input("Username", key="login_username")
            password = st.sidebar.text_input("Password", type="password", key="login_password")

            if st.sidebar.button("Login"):
                if verify_user(username, password):
                    st.session_state.token = generate_token(username)
                    st.session_state.username = username
                    st.sidebar.success("Logged in successfully!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials")

        # Landing Page for non-authenticated users
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://i.imgur.com/JqYeZvn.png", width=120)
        with col2:
            st.title("GlucoCheck Pro+")
            st.caption("Advanced Diabetes Management Platform")

        st.markdown("---")
        st.header("Please login or register to access all features")
        st.markdown("""
        - 🔒 Secure user accounts
        - 📈 Personalized health tracking
        - 🏆 Premium features after login
        """)

if __name__ == "__main__":
    main()
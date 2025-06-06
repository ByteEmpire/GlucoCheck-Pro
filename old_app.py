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

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'auth'))

from auth_utils import (
    initialize_user_db, hash_password, register_user, verify_user,
    generate_token, verify_token, update_password, delete_user
)

# Page Config
st.set_page_config(
    page_title="GlucoCheck Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
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

    .stButton>button:hover {
        background: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(108, 99, 255, 0.3);
    }

    .sidebar .sidebar-content {
        background: white;
        box-shadow: 5px 0 15px rgba(0,0,0,0.05);
    }

    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        padding: 10px 15px;
        border: 1px solid #e0e0e0;
    }

    .stSlider>div>div>div>div {
        background: var(--primary) !important;
    }

    .risk-card {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid {color};
        transition: all 0.3s;
    }

    .risk-card:hover {
        transform: translateY(-5px);
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

# Load Model
@st.cache_resource # Changed to st.cache_resource for models/scalers
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Personalized Recommendations Engine
def get_personalized_recommendations(probability, bmi, age):
    """5-tier recommendation system with personalized plans"""

    # Risk Classification
    if probability < 0.2:
        risk_level = "Very Low"
        color = "var(--success)"
        icon = "✅"
    elif 0.2 <= probability < 0.4:
        risk_level = "Low"
        color = "var(--success)"
        icon = "🟢"
    elif 0.4 <= probability < 0.6:
        risk_level = "Moderate"
        color = "var(--warning)"
        icon = "⚠️"
    elif 0.6 <= probability < 0.8:
        risk_level = "High"
        color = "var(--danger)"
        icon = "🔴"
    else:
        risk_level = "Very High"
        color = "var(--danger)"
        icon = "❗"

    # BMI Classification
    bmi_category = ""
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 25:
        bmi_category = "Normal"
    elif 25 <= bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Age Group
    if age < 30:
        age_group = "young adult"
    elif 30 <= age < 50:
        age_group = "middle-aged adult"
    else:
        age_group = "senior"

    # Dynamic Recommendations
    diet_plan = {
        "Very Low": [
            f"🍏 Maintain your healthy {age_group} diet",
            "💧 Stay hydrated with 2L water daily",
            "🌰 Include nuts and seeds for micronutrients"
        ],
        "Low": [
            f"🥗 Focus on fiber-rich foods as a {age_group}",
            "🍗 Lean protein sources 3x/day",
            "❌ Limit processed sugars"
        ],
        "Moderate": [
            f"⚖️ {bmi_category}-appropriate calorie intake",
            "🕒 Consistent meal timing",
            "🍎 Low-glycemic fruits preferred"
        ],
        "High": [
            f"⚠️ Strict carb control for {bmi_category} individuals",
            "🥦 5+ vegetable servings daily",
            "🐟 Omega-3 rich foods 4x/week"
        ],
        "Very High": [
            f"🚨 Medical nutrition therapy advised for {bmi_category} {age_group}",
            "🩺 Consult dietitian for personalized plan",
            "⏲️ Track all macronutrients"
        ]
    }

    exercise_plan = {
        "Very Low": [
            "🏃‍♂️ 30 mins cardio 5x/week",
            "🧘 2x/week flexibility training"
        ],
        "Low": [
            "🏋️‍♂️ Strength training 3x/week",
            "🚶‍♂️ 8,000 steps daily"
        ],
        "Moderate": [
            "🏊‍♀️ 150 mins moderate exercise weekly",
            "🤸‍♂️ Include HIIT workouts"
        ],
        "High": [
            "🚴‍♂️ 45 mins cardio daily",
            "🏋️‍♂️ Resistance training 4x/week"
        ],
        "Very High": [
            "🩺 Supervised exercise program",
            "⏱️ 60+ mins activity daily"
        ]
    }

    supplements = {
        "Very Low": ["Multivitamin", "Vitamin D"],
        "Low": ["Magnesium", "Omega-3"],
        "Moderate": ["Berberine", "Cinnamon extract"],
        "High": ["Alpha-lipoic acid", "Chromium"],
        "Very High": ["Prescription supplements as needed"]
    }

    return {
        "risk_level": risk_level,
        "color": color,
        "icon": icon,
        "diet": diet_plan[risk_level],
        "exercise": exercise_plan[risk_level],
        "supplements": supplements[risk_level],
        "bmi_category": bmi_category,
        "age_group": age_group
    }

def create_gauge(probability, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#4CAF50"},
                {'range': [20, 40], 'color': "#8BC34A"},
                {'range': [40, 60], 'color': "#FFC107"},
                {'range': [60, 80], 'color': "#FF9800"},
                {'range': [80, 100], 'color': "#F44336"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        }
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=50, r=50, t=80, b=20),
        font={'family': "Poppins"}
    )
    return fig

# Risk Card
def risk_card(title, value, color, icon):
    with stylable_container(
        key=f"card_{title}",
        css_styles=f"""
        {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-left: 5px solid {color};
            transition: all 0.3s;
        }}
        """
    ):
        st.markdown(f"""
        <div class="fade-in">
            <h3 style="color: {color}; margin-bottom: 5px;">{icon} {title}</h3>
            <h1 style="margin-top: 0; color: #333;">{value}</h1>
        </div>
        """, unsafe_allow_html=True)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import base64

def generate_pdf_report(probability, recommendations, age, bmi, glucose, bp, skin_thickness, insulin, pedigree, pregnancies):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    c.drawString(30, 750, "📄 GlucoCheck Pro - Personalized Diabetes Risk Report")
    c.drawString(30, 735, "-" * 90)

    c.drawString(30, 710, f"👤 Age: {age}  |  Pregnancies: {pregnancies}")
    c.drawString(30, 695, f"🧪 Glucose: {glucose} mg/dL | Blood Pressure: {bp} mmHg")
    c.drawString(30, 680, f"📏 Skin Thickness: {skin_thickness} mm | Insulin: {insulin} μU/mL")
    c.drawString(30, 665, f"⚖️ BMI: {bmi} | Pedigree: {pedigree}")

    c.drawString(30, 640, f"🔬 Diabetes Probability: {probability*100:.1f}%")
    c.drawString(30, 625, f"🧭 Risk Level: {recommendations['risk_level']}")
    c.drawString(30, 610, f"📊 BMI Category: {recommendations['bmi_category']}")
    c.drawString(30, 595, f"🎯 Age Group: {recommendations['age_group'].title()}")

    c.drawString(30, 570, "🍽️ Diet Recommendations:")
    for i, item in enumerate(recommendations["diet"], start=1):
        c.drawString(50, 570 - i*15, f"- {item}")

    offset = 570 - len(recommendations["diet"])*15 - 20
    c.drawString(30, offset, "🏋️ Exercise Recommendations:")
    for i, item in enumerate(recommendations["exercise"], start=1):
        c.drawString(50, offset - i*15, f"- {item}")

    offset -= len(recommendations["exercise"])*15 + 20
    c.drawString(30, offset, "💊 Suggested Supplements:")
    for i, item in enumerate(recommendations["supplements"], start=1):
        c.drawString(50, offset - i*15, f"- {item}")

    c.drawString(30, 60, "🔖 Note: Always consult with a healthcare professional for clinical decisions.")

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
        

def app_main_content():
    """Main application content displayed after successful authentication."""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://c4.wallpaperflare.com/wallpaper/659/894/522/blood-sugar-chronic-diabetes-diabetes-mellitus-wallpaper-preview.jpg", width=120)
    with col2:
        st.title("GlucoCheck Pro")
        st.caption("Advanced Diabetes Risk Assessment with Personalized Health Plans")

    st.markdown("---")

    # Sidebar Input
    with st.sidebar:
        st.title("Patient Profile")
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
        # Prediction
        input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, pedigree, age]])
        input_scaled = scaler.transform(input_data)

        probability = model.predict_proba(input_scaled)[0][1] # Use scaled input for prediction
        recommendations = get_personalized_recommendations(probability, bmi, age)

        # Results Dashboard
        with st.container():
            st.success("Analysis complete! Here's your personalized health report:")
    
            pdf_bytes = generate_pdf_report(probability, recommendations, age, bmi, glucose, bp, skin_thickness, insulin, pedigree, pregnancies)
            st.download_button(
                label="📥 Download Report",
                data=pdf_bytes,
                file_name="glucocheck_report.pdf",
                mime="application/pdf",
                help="Click to download your personalized report"
            )
            


            # Risk Overview Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_card("Risk Level", recommendations["risk_level"], recommendations["color"], recommendations["icon"])
            with col2:
                risk_card("Probability", f"{probability*100:.1f}%", recommendations["color"], "📊")
            with col3:
                risk_card("BMI Category", recommendations["bmi_category"], "#6C63FF", "⚖️")

            st.markdown("---")

            # Main Content
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

            # Data Visualization
            st.markdown("---")
            st.header("Health Insights")

            tab1, tab2 = st.tabs(["Risk Factors", "Glucose Analysis"])

            with tab1:
                feat_imp = pd.DataFrame({
                    'Factor': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig = px.bar(feat_imp, x='Importance', y='Factor', orientation='h',
                             title='What Impacts Your Diabetes Risk Most?',
                             color='Importance',
                             color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                try:
                    df = pd.read_csv("enhanced_diabetes.csv")
                    fig = px.scatter(df, x="Glucose", y="BMI", color="Outcome",
                                     hover_data=["Age"],
                                     title="Glucose vs BMI Relationship",
                                     color_discrete_map={0: "#4CAF50", 1: "#F44336"},
                                     trendline="lowess")
                    st.plotly_chart(fig, use_container_width=True)
                except FileNotFoundError:
                    st.warning("`enhanced_diabetes.csv` not found. Cannot display Glucose Analysis.")

    else:
        # Landing Page when `submitted` is False
        with st.container():
            st.header("Welcome to GlucoCheck Pro")
            st.markdown("""
            **The most advanced diabetes risk assessment tool** that provides:
            - 🎯 Personalized risk scoring
            - 🍏 Custom nutrition plans
            - 🏋️‍♂️ Tailored exercise recommendations
            - 📊 Interactive health insights
            """)

            cols = st.columns(3)
            with cols[0]:
                card(
                    title="Advanced AI Model",
                    text="92% accurate predictions using XGBoost",
                    image="https://www.openaccessgovernment.org/wp-content/uploads/2024/03/iStock-1487972668-scaled.jpg"
                )
            with cols[1]:
                card(
                    title="Personalized Plans",
                    text="Custom recommendations based on your profile",
                    image="https://merakibydrdivya.com/wp-content/uploads/sites/163/2025/01/diabetes-diet-1.png"
                )
            with cols[2]:
                card(
                    title="Health Dashboard",
                    text="Interactive visualizations of your risk factors",
                    image="https://www.shutterstock.com/image-vector/blood-glucose-meter-isolated-concept-600nw-2466152465.jpg"
                )

            st.markdown("---")
            st.subheader("How It Works")
            steps = st.columns(3)
            with steps[0]:
                st.markdown("""
                ### 1️⃣ Enter Your Metrics
                Provide basic health information in the sidebar
                """)
            with steps[1]:
                st.markdown("""
                ### 2️⃣ Get Your Risk Analysis
                Our AI calculates your diabetes probability
                """)
            with steps[2]:
                st.markdown("""
                ### 3️⃣ Receive Your Plan
                Personalized diet, exercise and supplement recommendations
                """)

            st.markdown("""
            <div style="text-align: center; margin-top: 40px;">
                <h3>Ready to check your risk?</h3>
                <p>Fill out the form in the sidebar to begin</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    initialize_user_db() # Ensure the user database exists

    # Initialize session state variables for authentication
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'username' not in st.session_state:
        st.session_state.username = None

    # Verify token if one exists in session state
    verified, user_or_message = (False, None)
    if st.session_state.token:
        verified, user_or_message = verify_token(st.session_state.token)
        if verified:
            st.session_state.username = user_or_message
        else:
            st.session_state.token = None # Clear invalid or expired token
            st.session_state.username = None


    if st.session_state.username:
        # User is authenticated, show app content and logout button
        st.sidebar.success(f"Welcome, **{st.session_state.username}**!")
        if st.sidebar.button("Logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun() # <<< CHANGED: Use st.rerun()
        app_main_content() # Call function to display the main app features
    else:
        # User is not authenticated, show login/register options
        st.sidebar.header("Authentication")
        auth_option = st.sidebar.radio("Choose an option", ["Login", "Register"])

        if auth_option == "Register":
            st.sidebar.subheader("New User Registration")
            new_username = st.sidebar.text_input("New Username", key="reg_username")
            new_password = st.sidebar.text_input("New Password", type="password", key="reg_password")
            confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="reg_confirm_password")

            if st.sidebar.button("Register Account"):
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        if register_user(new_username, new_password):
                            st.sidebar.success("Registration successful! Please log in.")
                        else:
                            st.sidebar.error("Username already exists. Try a different one.")
                    else:
                        st.sidebar.error("Passwords do not match.")
                else:
                    st.sidebar.error("Please fill in all fields.")
        else: # Login
            st.sidebar.subheader("User Login")
            username = st.sidebar.text_input("Username", key="login_username")
            password = st.sidebar.text_input("Password", type="password", key="login_password")

            if st.sidebar.button("Login"):
                if username and password:
                    if verify_user(username, password):
                        st.session_state.token = generate_token(username)
                        st.session_state.username = username
                        st.sidebar.success("Logged in successfully!")
                        time.sleep(0.5) # Give a moment for message to show
                        st.rerun() # <<< CHANGED: Use st.rerun()
                    else:
                        st.sidebar.error("Invalid username or password.")
                else:
                    st.sidebar.error("Please enter both username and password.")

        # Display the landing page when not authenticated, or after logout/failed login
        # This part ensures the initial landing page is visible before login.
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://c4.wallpaperflare.com/wallpaper/659/894/522/blood-sugar-chronic-diabetes-diabetes-mellitus-wallpaper-preview.jpg", width=120)
        with col2:
            st.title("GlucoCheck Pro")
            st.caption("Advanced Diabetes Risk Assessment with Personalized Health Plans")

        st.markdown("---")

        with st.container():
            st.header("Welcome to GlucoCheck Pro")
            st.markdown("""
            **The most advanced diabetes risk assessment tool** that provides:
            - 🎯 Personalized risk scoring
            - 🍏 Custom nutrition plans
            - 🏋️‍♂️ Tailored exercise recommendations
            - 📊 Interactive health insights
            """)

            cols = st.columns(3)
            with cols[0]:
                card(
                    title="Advanced AI Model",
                    text="92% accurate predictions using XGBoost",
                    image="https://www.openaccessgovernment.org/wp-content/uploads/2024/03/iStock-1487972668-scaled.jpg"
                )
            with cols[1]:
                card(
                    title="Personalized Plans",
                    text="Custom recommendations based on your profile",
                    image="https://merakibydrdivya.com/wp-content/uploads/sites/163/2025/01/diabetes-diet-1.png"
                )
            with cols[2]:
                card(
                    title="Health Dashboard",
                    text="Interactive visualizations of your risk factors",
                    image="https://www.shutterstock.com/image-vector/blood-glucose-meter-isolated-concept-600nw-2466152465.jpg"
                )

            st.markdown("---")
            st.subheader("How It Works")
            steps = st.columns(3)
            with steps[0]:
                st.markdown("""
                ### 1️⃣ Enter Your Metrics
                Provide basic health information in the sidebar
                """)
            with steps[1]:
                st.markdown("""
                ### 2️⃣ Get Your Risk Analysis
                Our AI calculates your diabetes probability
                """)
            with steps[2]:
                st.markdown("""
                ### 3️⃣ Receive Your Plan
                Personalized diet, exercise and supplement recommendations
                """)

            st.markdown("""
            <div style="text-align: center; margin-top: 40px;">
                <h3>Ready to check your risk?</h3>
                <p>Login or Register in the sidebar to begin</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    
# doctor code -

def ai_doctor_chat():
    st.header("🩺 AI Diabetes Specialist")
    st.caption("Get answers to your diabetes-related questions (not a substitute for medical advice)")
    
    assistant = AIDoctorAssistant()
    
    # Display chat history
    for message in st.session_state.chat_history:
        # Use the full, granular timestamp for the key to ensure uniqueness
        # And format it nicely for display
        unique_key_timestamp = message['timestamp'] # This will be the granular timestamp string
        display_timestamp = datetime.datetime.strptime(unique_key_timestamp, "%Y%m%d%H%M%S%f").strftime("%H:%M")

        with stylable_container(
            key=f"chat_{unique_key_timestamp}", # This key will now be unique
            css_styles=f"""
            {{
                border-radius: 18px;
                padding: 12px 16px;
                margin: 8px 0;
                max-width: 80%;
                background-color: {'#6C63FF' if message['role'] == 'user' else '#f0f2f6'};
                color: {'white' if message['role'] == 'user' else 'inherit'};
                margin-left: {'auto' if message['role'] == 'user' else '0'};
                margin-right: {'0' if message['role'] == 'user' else 'auto'};
            }}
            """
        ):
            st.markdown(f"**{'You' if message['role'] == 'user' else 'AI Doctor'}**: {message['content']}")
            st.caption(display_timestamp) # Display the user-friendly formatted time
    
    # Chat input
    user_query = st.chat_input("Ask your diabetes-related question...")
    if user_query:
        # Generate a highly granular timestamp including microseconds
        granular_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": granular_timestamp # Store the granular timestamp
        })
        
        # Generate context from current health data
        context = ""
        if st.session_state.health_history:
            latest = st.session_state.health_history[-1]
            context = f"User's latest health stats: {latest['risk_level']} risk, BMI {latest['bmi']}, Glucose {latest['glucose']}, Age {latest['age']}"
        
        # Get AI response
        with st.spinner("AI Doctor is thinking..."):
            ai_response = assistant.generate_response(user_query, context)
        
        # Generate another highly granular timestamp for AI response
        granular_timestamp_ai = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # Add AI response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": granular_timestamp_ai # Store the granular timestamp
        })
        st.rerun()
        
        
# NEWER CODE -

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

# --- Path Setup ---
current_dir = os.path.dirname(__file__)
# Ensure the 'auth' directory is correctly recognized.
# This assumes 'auth_utils.py' is in a folder named 'auth' in the same directory as this script.
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
    st.error("Authentication utilities (auth_utils.py) not found. Please ensure the 'auth' folder and 'auth_utils.py' are correctly placed.")
    # Define dummy functions to prevent immediate crashes if auth_utils is missing
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
    
    .chat-bubble {
        border-radius: 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-bubble {
        background-color: #6C63FF;
        color: white;
        margin-left: auto;
    }
    
    .ai-bubble {
        background-color: #f0f2f6;
        margin-right: auto;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .medication-card {
        border-left: 4px solid #6C63FF;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# --- AI Doctor Chat Class ---
class AIDoctorAssistant:
    def __init__(self):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            
            # Recommended models (try in this order)
            model_options = [
                'models/gemini-1.5-flash',
                'models/gemini-1.5-flash-latest'
                'models/gemini-pro',
                'models/gemini-pro-vision'
            ]
            
            # Find the first available model
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
        
        self.system_prompt = """..."""  # Your existing prompt

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

init_session_state()

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Ensure model and scaler files exist in the same directory as the app script.
    model_path = os.path.join(current_dir, "model.pkl")
    scaler_path = os.path.join(current_dir, "scaler.pkl")

    model = None
    scaler = None
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.warning(f"Model file not found: {model_path}. Risk assessment will not function.")
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            st.warning(f"Scaler file not found: {scaler_path}. Risk assessment will not function correctly.")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
    return model, scaler

model, scaler = load_model()

# --- Core Functions (Existing) ---
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
        color = "#4CAF50"  # Green
        icon = "😊"
        diet = ["Maintain balanced diet with whole foods.", "Limit processed sugars and unhealthy fats.", "Stay hydrated."]
        exercise = ["Regular moderate activity (e.g., brisk walking 30 min, 5 times/week).", "Incorporate strength training 2-3 times/week."]
        supplements = ["Multivitamin (if diet lacks variety).", "Omega-3 fatty acids."]
    elif 0.2 <= probability < 0.4:
        risk_level = "Low"
        color = "#8BC34A"  # Light Green
        icon = "🙂"
        diet = ["Focus on portion control.", "Increase fiber intake (fruits, vegetables, whole grains).", "Reduce sugary drinks."]
        exercise = ["Aim for 150 minutes of moderate-intensity exercise weekly.", "Try activities like cycling, swimming, or dancing."]
        supplements = ["Vitamin D (especially if limited sun exposure)."]
    elif 0.4 <= probability < 0.6:
        risk_level = "Moderate"
        color = "#FFC107"  # Amber
        icon = "😐"
        diet = ["Strictly limit refined carbs and added sugars.", "Emphasize lean proteins and healthy fats.", "Consider consulting a nutritionist."]
        exercise = ["Increase physical activity to 200-250 minutes of moderate intensity per week.", "Include both cardio and strength exercises."]
        supplements = ["Chromium picolinate (may help with insulin sensitivity).", "Magnesium."]
    elif 0.6 <= probability < 0.8:
        risk_level = "High"
        color = "#FF9800"  # Orange
        icon = "😟"
        diet = ["Adopt a low-glycemic index diet.", "Eliminate sugary snacks and drinks.", "Work with a dietitian for a personalized meal plan."]
        exercise = ["Intensify exercise to 250-300 minutes of moderate activity, or 150 minutes of vigorous activity.", "Monitor blood sugar before/after exercise."]
        supplements = ["Berberine (natural compound with glucose-lowering effects, consult doctor).", "Alpha-lipoic acid."]
    else:
        risk_level = "Very High"
        color = "#F44336"  # Red
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
                {'range': [0, 20], 'color': "#4CAF50"},  # Very Low
                {'range': [20, 40], 'color': "#8BC34A"},  # Low
                {'range': [40, 60], 'color': "#FFC107"},  # Moderate
                {'range': [60, 80], 'color': "#FF9800"},  # High
                {'range': [80, 100], 'color': "#F44336"}  # Very High
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

# --- New Feature: AI Doctor Chat ---
# --- New Feature: AI Doctor Chat ---
def ai_doctor_chat():
    st.header("🩺 AI Diabetes Specialist")
    st.caption("Get answers to your diabetes-related questions (not a substitute for medical advice)")
    
    assistant = AIDoctorAssistant()
    
    # Main chat container with custom styling
    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)
    
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
    
    # Display the chat
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # Chat input
    user_query = st.chat_input("Ask your diabetes-related question...")
    if user_query:
        # Generate timestamp
        granular_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": granular_timestamp
        })
        
        # Generate context
        context = ""
        if st.session_state.health_history:
            latest = st.session_state.health_history[-1]
            context = f"User's latest health stats: {latest['risk_level']} risk, BMI {latest['bmi']}, Glucose {latest['glucose']}, Age {latest['age']}"
        
        # Get AI response
        with st.spinner("AI Doctor is thinking..."):
            ai_response = assistant.generate_response(user_query, context)
        
        # Add AI response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        })
        st.rerun()

# --- New Feature: Health Timeline ---
def display_health_timeline():
    st.header("📈 Your Health Timeline")
    
    if not st.session_state.health_history:
        st.info("No health assessments recorded yet. Complete an assessment to see your timeline.")
        return
    
    # Create DataFrame for visualization
    timeline_data = pd.DataFrame(st.session_state.health_history)
    timeline_data['date'] = pd.to_datetime(timeline_data['date'])
    timeline_data.sort_values('date', inplace=True)
    
    # Plotly timeline visualization
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
    
    # Detailed history
    with st.expander("📝 View Detailed History"):
        for entry in reversed(st.session_state.health_history):
            with stylable_container(
                key=f"entry_{entry['date']}",
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

# --- New Feature: Medication Planner ---
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
        
        if st.form_submit_button("➕ Add Medication"):
            if name and dosage and time_of_day:
                new_med = {
                    "name": name,
                    "dosage": dosage,
                    "frequency": frequency,
                    "times": time_of_day,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "reminders": True
                }
                st.session_state.medications.append(new_med)
                st.success("Medication added successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields")
    
    # Current medications list
    st.subheader("Your Medications")
    if not st.session_state.medications:
        st.info("No medications added yet.")
    else:
        for i, med in enumerate(st.session_state.medications):
            with stylable_container(
                key=f"med_{i}",
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
                with cols[1]:
                    if st.button("❌ Remove", key=f"remove_{i}"):
                        st.session_state.medications.pop(i)
                        st.rerun()
    
    # Calendar view
    st.subheader("📅 Medication Schedule")
    if st.session_state.medications:
        # Generate calendar events
        events = []
        colors = ["#6C63FF", "#4D44DB", "#FF6B6B", "#4CAF50", "#FFC107"]
        
        for i, med in enumerate(st.session_state.medications):
            for time_period in med['times']:
                # For calendar display, we'll assign a fixed time for each period today
                start_time_str = get_time_for_period(time_period, end=False)
                end_time_str = get_time_for_period(time_period, end=True)
                
                # Create events for a week, or adjust logic for full scheduling
                for d_offset in range(7): # Show for 7 days
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
            "initialView": "timeGridWeek", # Or "dayGridMonth"
            "slotMinTime": "06:00:00",
            "slotMaxTime": "22:00:00",
            "navLinks": True,
            "selectable": True,
            "editable": False, # Prevent drag/drop for simplicity
            "dayMaxEvents": True, # display "more" link if too many events
            "height": "auto", # Adjust height dynamically
            "aspectRatio": 2 # Control aspect ratio
        }
        
        # The calendar component needs a unique key if it's within a loop or prone to re-creation
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
    """Main application content displayed after successful authentication."""
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
            st.error("Cannot perform risk assessment: Model or scaler not loaded. Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory.")
            return

        # Prediction
        input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, pedigree, age]])
        input_scaled = scaler.transform(input_data)

        probability = model.predict_proba(input_scaled)[0][1]
        recommendations = get_personalized_recommendations(probability, bmi, age)

        # Save to health history
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

        # Results Dashboard
        with st.container():
            st.success("Analysis complete! Here's your personalized health report:")
            
            # PDF Report
            pdf_bytes = generate_pdf_report(probability, recommendations, age, bmi, glucose, bp, skin_thickness, insulin, pedigree, pregnancies)
            st.download_button(
                label="📥 Download Full Report (PDF)",
                data=pdf_bytes,
                file_name=f"glucocheck_report_{datetime.date.today()}.pdf",
                mime="application/pdf"
            )

            # Risk Overview Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_card("Risk Level", recommendations["risk_level"], recommendations["color"], recommendations["icon"])
            with col2:
                risk_card("Probability", f"{probability*100:.1f}%", recommendations["color"], "📊")
            with col3:
                risk_card("BMI Category", recommendations["bmi_category"], "#6C63FF", "⚖️")

            st.markdown("---")

            # Main Content
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

        # Data Visualization
        st.markdown("---")
        st.header("Health Insights")

        tab1, tab2 = st.tabs(["Risk Factors", "Glucose Analysis"])

        with tab1:
            if model is not None and hasattr(model, 'feature_importances_'): # Check if model has feature_importances_
                # The order of features in feature_importances_ must match the order
                # of input features used during model training.
                # Assuming the order is: pregnancies, glucose, bp, skin_thickness, insulin, bmi, pedigree, age
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
                st.warning("Model not loaded or does not support feature importance calculation.")
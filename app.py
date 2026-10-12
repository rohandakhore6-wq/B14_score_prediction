import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS WITH SHADOWS AND PROFESSIONAL UI ---
st.markdown("""
    <style>
        /* Main Background & Font */
        .main {
            background-color: #f8f9fa;
            font-family: 'Inter', sans-serif;
        }
        
        /* Card Container Styling with Shadows */
        .card-container {
            background: #ffffff;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
        }
        
        /* Header Styling */
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        /* Metric Box Styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Custom Button Styling */
        .stButton>button {
            width: 100%;
            background-color: #667eea;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #5a6fd6;
            box-shadow: 0 6px 10px rgba(102, 126, 234, 0.3);
            transform: translateY(-1px);
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL CACHE ---
@st.cache_resource
def load_model():
    model_path = "kmn.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# --- SIDEBAR INFO ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/student-center.png", width=120)
    st.title("Model Dashboard")
    st.markdown("---")
    st.info(
        "**About this app:**\n\n"
        "This application uses a K-Nearest Neighbors Regressor (`kmn.pkl`) "
        "to predict student outcomes based on academic habits, attendance, "
        "and lifestyle parameters."
    )
    st.markdown("### Instructions")
    st.markdown("1. Adjust the student parameters on the main panel.\n2. Click **Predict Performance** to run the model.")

# --- MAIN INTERFACE ---
st.title("🎓 Student Academic Performance Predictor")
st.markdown("Provide the student's details below to estimate performance outcomes using the deployed machine learning model.")

if model is None:
    st.error("⚠️ Model file `kmn.pkl` not found in the root directory! Please ensure it is uploaded correctly.")
else:
    # Form Container for Inputs
    with st.container():
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader("📋 Student Profile & Study Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", min_value=15, max_value=40, value=20, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            course = st.selectbox("Course / Stream", options=["Computer Science", "Engineering", "Business", "Arts", "Medicine", "Science"])
            study_hours = st.number_input("Daily Study Hours", min_value=0.0, max_value=24.0, value=4.5, step=0.5)

        with col2:
            class_attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, value=85, step=1)
            internet_access = st.selectbox("Internet Access", options=["Yes", "No"])
            sleep_hours = st.slider("Sleep Hours per Night", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
            sleep_quality = st.slider("Sleep Quality Rating (1-10)", min_value=1, max_value=10, value=7)

        with col3:
            study_method = st.selectbox("Primary Study Method", options=["Self-study", "Online Courses", "Group Study", "Textbooks"])
            facility_rating = st.slider("Campus Facility Rating (1-5)", min_value=1, max_value=5, value=4)
            exam_difficulty = st.slider("Perceived Exam Difficulty (1-5)", min_value=1, max_value=5, value=3)

        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Action
    col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
    with col_center2:
        predict_btn = st.button("🚀 Predict Performance")

    if predict_btn:
        # Preprocessing / Mapping user inputs to model features
        # Note: Adjust categorical mappings if your training pipeline used specific one-hot encoding or LabelEncoding.
        gender_encoded = 1 if gender == "Female" else (0 if gender == "Male" else 2)
        internet_encoded = 1 if internet_access == "Yes" else 0
        
        # Simple categorical encoding placeholders mapping common categories
        course_mapping = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts": 3, "Medicine": 4, "Science": 5}
        method_mapping = {"Self-study": 0, "Online Courses": 1, "Group Study": 2, "Textbooks": 3}
        
        course_encoded = course_mapping.get(course, 0)
        method_encoded = study_method.get(study_method, 0) if isinstance(study_method, dict) else method_mapping.get(study_method, 0)

        # Assemble input feature array matching the model's expected feature list:
        # ['age', 'gender', 'course', 'study_hours', 'class_attendance', 'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
        input_data = np.array([[
            age, 
            gender_encoded, 
            course_encoded, 
            study_hours, 
            class_attendance, 
            internet_access, # or internet_encoded based on your preprocessing training setup
            sleep_hours, 
            sleep_quality, 
            method_encoded, 
            facility_rating, 
            exam_difficulty
        ]])

        try:
            # Make prediction
            prediction = model.predict(input_data)
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>Estimated Performance Score</h3>
                        <h1 style="color: white; font-size: 2.5rem;">{prediction[0]:.2f}</h1>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with res_col2:
                st.success("Prediction generated successfully using KNeighborsRegressor!")
                st.write(f"- **Model Type:** {type(model).__name__}")
                st.write("- **Confidence / Status:** Optimal parameters evaluated.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Tip: Ensure that categorical variables match the exact encoding format used when training the model.")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from streamlit_lottie import st_lottie
import requests

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="E-commerce Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UI/UX ---
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #0e1117;
    }
    
    /* Custom Card Style */
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    
    /* Text Styles */
    h1 { color: #ffffff !important; font-family: 'Inter', sans-serif; }
    h3 { color: #a1a1a1 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILS ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_searching = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_at6m82it.json")
lottie_success = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_atippmse.json")
lottie_fail = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ghp9v6m9.json")

@st.cache_resource
def load_models():
    # Placeholder for your local files
    try:
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')
        knn = joblib.load('knn_model.pkl')
        return scaler, pca, knn
    except:
        st.error("Model files not found. Please ensure .pkl files are in the directory.")
        return None, None, None

scaler, pca, knn = load_models()

# --- SIDEBAR UI ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
st.sidebar.title("Configuration")

with st.sidebar.expander("📄 Page Browsing Metrics", expanded=True):
    Admin = st.slider('Administrative Pages', 0, 30, 2)
    Admin_Dur = st.number_input('Admin Duration (mins)', 0.0, 5000.0, 45.0)
    Info = st.slider('Informational Pages', 0, 30, 1)
    Info_Dur = st.number_input('Info Duration (mins)', 0.0, 3000.0, 10.0)
    Prod = st.slider('Product Pages', 0, 700, 25)
    Prod_Dur = st.number_input('Product Duration (mins)', 0.0, 60000.0, 500.0)

with st.sidebar.expander("📊 Session Analytics"):
    Bounce = st.slider('Bounce Rate', 0.0, 0.2, 0.01)
    Exit = st.slider('Exit Rate', 0.0, 0.2, 0.02)
    PageVal = st.slider('Page Value', 0.0, 400.0, 25.0)
    SpecialDay = st.slider('Special Day Proximity', 0.0, 1.0, 0.0)

with st.sidebar.expander("🌍 Contextual Data"):
    Month = st.selectbox('Month', options=['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    VisitorType = st.selectbox('Visitor Type', options=['Returning_Visitor', 'New_Visitor', 'Other'])
    Weekend = st.checkbox('Is it a Weekend?')

# --- DATA PROCESSING ---
month_map = {'Feb': 2, 'Mar': 5, 'May': 6, 'June': 4, 'Jul': 3, 'Aug': 0, 'Sep': 9, 'Oct': 8, 'Nov': 7, 'Dec': 1}
visitor_map = {'New_Visitor': 0, 'Other': 1, 'Returning_Visitor': 2}

data = {
    'Administrative': Admin, 'Administrative_Duration': Admin_Dur,
    'Informational': Info, 'Informational_Duration': Info_Dur,
    'ProductRelated': Prod, 'ProductRelated_Duration': Prod_Dur,
    'BounceRates': Bounce, 'ExitRates': Exit, 'PageValues': PageVal,
    'SpecialDay': SpecialDay, 'Month': month_map[Month],
    'OperatingSystems': 2, 'Browser': 2, 'Region': 1, 'TrafficType': 2,
    'VisitorType': visitor_map[VisitorType], 'Weekend': int(Weekend)
}
input_df = pd.DataFrame(data, index=[0])

# --- MAIN CONTENT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🛍️ E-commerce Behavior Intelligence")
    st.write("### AI-Driven Purchase Intent Prediction")
    st.markdown(f"""
    **Lead Researcher:** Sahil Singh  
    **Framework:** KNN + PCA + SMOTE Oversampling
    """)
    
    st.markdown("---")
    
    # Display Metrics in modern cards
    st.subheader("Current Session Snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pages Visited", f"{Admin + Info + Prod}")
    m2.metric("Total Duration", f"{round(Admin_Dur + Info_Dur + Prod_Dur, 1)}m")
    m3.metric("Bounce Rate", f"{Bounce:.2%}")
    m4.metric("Page Value", f"${PageVal}")

with col2:
    if lottie_searching:
        st_lottie(lottie_searching, height=250, key="search")

# --- PREDICTION LOGIC ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 ANALYZE BEHAVIOR"):
    if scaler and pca and knn:
        with st.spinner("Processing High-Dimensional Data..."):
            # Simulation of compute time for UX
            time.sleep(1.5)
            
            scaled_data = scaler.transform(input_df)
            pca_data = pca.transform(scaled_data)
            prediction = knn.predict(pca_data)
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediction[0] == 1:
                    st_lottie(lottie_success, height=200)
                else:
                    st_lottie(lottie_fail, height=200)
            
            with res_col2:
                if prediction[0] == 1:
                    st.success("## 🎯 High Conversion Intent")
                    st.write("This user shows strong behavioral signals for revenue generation. Consider offering a time-limited discount to close the sale.")
                else:
                    st.error("## 📉 Low Conversion Intent")
                    st.write("The browsing patterns suggest this session is likely for research or window shopping. Retargeting may be required.")
    else:
        st.warning("Model components are missing. Check your 'load_models' function.")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<center style='color: #666;'>Geeta University | Department of CSE (AI & ML)</center>", unsafe_allow_html=True)
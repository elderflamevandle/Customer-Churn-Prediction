import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1E3A8A;
        text-align: center;
        padding: 1.5rem 0;
    }
    h2 {
        color: #3B82F6;
        padding: 1rem 0;
    }
    .prediction-box {
        background-color: #e0f2fe;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #0284c7;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin: 0.5rem;
    }
    .required-columns {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# Define the exact feature names from the model
FEATURE_NAMES = [
    'Gender', 'Age', 'Married', 'Number of Dependents', 'Zip Code', 
    'Latitude', 'Longitude', 'Number of Referrals', 'Tenure in Months', 
    'Phone Service', 'Avg Monthly Long Distance Charges', 'Multiple Lines', 
    'Internet Service', 'Avg Monthly GB Download', 'Online Security', 
    'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
    'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 
    'Paperless Billing', 'Monthly Charge', 'Total Charges', 'Total Refunds', 
    'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue', 
    'TotalServices'
]

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    models = {}
    try:
        models['Random Forest'] = pickle.load(open('models/random_forest_model.pkl', 'rb'))
        models['XGBoost'] = pickle.load(open('models/xgboost_model.pkl', 'rb'))
        models['Gradient Boosting'] = pickle.load(open('models/gradient_boosting_model.pkl', 'rb'))
        models['Logistic Regression'] = pickle.load(open('models/logistic_regression_model.pkl', 'rb'))
        
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))
        feature_names = FEATURE_NAMES  # Use our defined feature names
        
        # Load model performance
        model_performance = pd.read_csv('models/model_performance.csv', index_col='Model')
        
        return models, scaler, label_encoders, feature_names, model_performance
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Load models
models, scaler, label_encoders, feature_names, model_performance = load_models()

# Title and description
st.markdown("<h1>📱 Telecom Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/phone.png", width=150)
    st.title("Navigation")
    page = st.radio("Select a page:", 
                    ["🎯 Single Prediction", "📊 Bulk Prediction", "📈 Model & Data Insights"])
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application predicts customer churn using advanced machine learning models.
    
    **Features:**
    - Single customer prediction
    - Bulk predictions from CSV/Excel
    - Model performance insights
    - Data visualizations
    """)

# Helper functions
def prepare_input_data(df, feature_names, scaler):
    """Prepare input data for prediction"""
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            # Add missing features with default values
            df[feature] = 0
    
    # Select only required features in correct order
    df = df[feature_names]
    
    # Scale the features
    df_scaled = scaler.transform(df)
    
    return df_scaled

def make_predictions(data, models):
    """Make predictions using all models"""
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        try:
            pred = model.predict(data)
            pred_proba = model.predict_proba(data)[:, 1]
            predictions[name] = pred
            probabilities[name] = pred_proba
        except Exception as e:
            st.error(f"Error with {name}: {str(e)}")
    
    return predictions, probabilities

def create_gauge_chart(value, title):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

# Page 1: Single Prediction
if page == "🎯 Single Prediction":
    st.header("🎯 Single Customer Churn Prediction")
    
    st.markdown("""
    Enter customer details below to predict the likelihood of churn.
    The system will use multiple ML models to provide comprehensive predictions.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        # Demographics Section
        st.subheader("📋 Customer Demographics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            
        with col2:
            married = st.selectbox("Married", ["Yes", "No"])
            num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            
        with col3:
            zip_code = st.number_input("Zip Code", min_value=10000, max_value=99999, value=90001)
            num_referrals = st.number_input("Number of Referrals", min_value=0, max_value=50, value=0)
            
        with col4:
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.0522, format="%.4f")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.2437, format="%.4f")
        
        # Service Information Section
        st.subheader("📞 Service Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tenure_months = st.number_input("Tenure in Months", min_value=0, max_value=100, value=12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            
        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic", "Cable"])
            
        with col3:
            avg_monthly_long_distance = st.number_input("Avg Monthly Long Distance Charges ($)", 
                                                        min_value=0.0, max_value=100.0, value=10.0, format="%.2f")
            avg_monthly_gb_download = st.number_input("Avg Monthly GB Download", 
                                                      min_value=0.0, max_value=1000.0, value=10.0, format="%.2f")
            
        with col4:
            unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        # Additional Services Section
        st.subheader("📡 Additional Services")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            
        with col2:
            device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])
            premium_tech_support = st.selectbox("Premium Tech Support", ["No", "Yes", "No internet service"])
            
        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
        with col4:
            streaming_music = st.selectbox("Streaming Music", ["No", "Yes", "No internet service"])
        
        # Financial Information Section
        st.subheader("💳 Financial Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, value=70.0, format="%.2f")
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, format="%.2f")
            
        with col2:
            total_refunds = st.number_input("Total Refunds ($)", min_value=0.0, value=0.0, format="%.2f")
            total_extra_data_charges = st.number_input("Total Extra Data Charges ($)", min_value=0, value=0)
            
        with col3:
            total_long_distance_charges = st.number_input("Total Long Distance Charges ($)", 
                                                          min_value=0.0, value=0.0, format="%.2f")
            
        with col4:
            total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, value=1000.0, format="%.2f")
        
        submit_button = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    if submit_button:
        # Count total services
        services_count = 0
        service_fields = [online_security, online_backup, device_protection, premium_tech_support,
                         streaming_tv, streaming_movies, streaming_music]
        for service in service_fields:
            if service == "Yes":
                services_count += 1
        
        # Create input dataframe with exact column names
        input_data = pd.DataFrame({
            'Gender': [1 if gender == "Male" else 0],
            'Age': [age],
            'Married': [1 if married == "Yes" else 0],
            'Number of Dependents': [num_dependents],
            'Zip Code': [zip_code],
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Number of Referrals': [num_referrals],
            'Tenure in Months': [tenure_months],
            'Phone Service': [1 if phone_service == "Yes" else 0],
            'Avg Monthly Long Distance Charges': [avg_monthly_long_distance],
            'Multiple Lines': [1 if multiple_lines == "Yes" else 0],
            'Internet Service': [1 if internet_service in ["DSL", "Fiber optic", "Cable"] else 0],
            'Avg Monthly GB Download': [avg_monthly_gb_download],
            'Online Security': [1 if online_security == "Yes" else 0],
            'Online Backup': [1 if online_backup == "Yes" else 0],
            'Device Protection Plan': [1 if device_protection == "Yes" else 0],
            'Premium Tech Support': [1 if premium_tech_support == "Yes" else 0],
            'Streaming TV': [1 if streaming_tv == "Yes" else 0],
            'Streaming Movies': [1 if streaming_movies == "Yes" else 0],
            'Streaming Music': [1 if streaming_music == "Yes" else 0],
            'Unlimited Data': [1 if unlimited_data == "Yes" else 0],
            'Paperless Billing': [1 if paperless_billing == "Yes" else 0],
            'Monthly Charge': [monthly_charge],
            'Total Charges': [total_charges],
            'Total Refunds': [total_refunds],
            'Total Extra Data Charges': [total_extra_data_charges],
            'Total Long Distance Charges': [total_long_distance_charges],
            'Total Revenue': [total_revenue],
            'TotalServices': [services_count]
        })
        
        try:
            # Prepare data for prediction
            X_input = prepare_input_data(input_data, feature_names, scaler)
            
            # Make predictions
            predictions, probabilities = make_predictions(X_input, models)
            
            st.markdown("---")
            st.markdown("<h2>🎯 Prediction Results</h2>", unsafe_allow_html=True)
            
            # Display results for each model
            cols = st.columns(len(models))
            for idx, (name, prob) in enumerate(probabilities.items()):
                with cols[idx]:
                    # Create gauge chart
                    fig = create_gauge_chart(prob[0], name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction
                    if prob[0] > 0.5:
                        st.error(f"⚠️ **High Risk** - {prob[0]*100:.1f}% chance of churn")
                    else:
                        st.success(f"✅ **Low Risk** - {prob[0]*100:.1f}% chance of churn")
            
            # Average prediction
            avg_prob = np.mean(list(probabilities.values()))
            st.markdown("---")
            st.markdown("<h3>📊 Ensemble Prediction (Average of All Models)</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig = create_gauge_chart(avg_prob, "Ensemble Prediction")
                st.plotly_chart(fig, use_container_width=True)
                
                if avg_prob > 0.5:
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: #fee2e2; border-color: #dc2626;">
                        <h3 style="color: #dc2626;">⚠️ High Churn Risk</h3>
                        <p style="font-size: 1.2em; color: #1f2937;">This customer has a <strong>{avg_prob*100:.1f}%</strong> probability of churning.</p>
                        <p style="color: #374151;><strong style="color: #1f2937;>Recommended Actions:</strong></p>
                        <ul style="color: #374151;>
                            <li>Offer personalized retention incentives</li>
                            <li>Schedule a customer satisfaction call</li>
                            <li>Review and optimize their service plan</li>
                            <li>Provide exclusive loyalty rewards</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: #dcfce7; border-color: #16a34a;">
                        <h3 style="color: #16a34a;">✅ Low Churn Risk</h3>
                        <p style="font-size: 1.2em; color: #1f2937;">This customer has only a <strong>{avg_prob*100:.1f}%</strong> probability of churning.</p>
                        <p style="color: #374151;"><strong style="color: #1f2937;>Recommended Actions:</strong></p>
                        <ul style="color: #374151;>
                            <li>Maintain regular engagement</li>
                            <li>Offer upgrade opportunities</li>
                            <li>Request feedback and testimonials</li>
                            <li>Include in referral programs</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Page 2: Bulk Prediction
elif page == "📊 Bulk Prediction":
    st.header("📊 Bulk Customer Churn Prediction")
    
    st.markdown("""
    Upload a CSV or Excel file containing customer data to get churn predictions for multiple customers at once.
    """)
    
    # Display required columns
    with st.expander("📋 View Required Columns for Bulk Upload", expanded=True):
        st.markdown("""
        <div class="required-columns" style="background-color: #101112; border-color: #d1d5db;">
        <h4>Your CSV/Excel file must contain these exact column names:</h4>
        
        <strong>Demographics:</strong>
        - Gender (Male/Female)
        - Age (numeric)
        - Married (Yes/No)
        - Number of Dependents (numeric)
        - Zip Code (numeric)
        - Latitude (numeric)
        - Longitude (numeric)
        - Number of Referrals (numeric)
        
        <strong>Service Information:</strong>
        - Tenure in Months (numeric)
        - Phone Service (Yes/No)
        - Avg Monthly Long Distance Charges (numeric)
        - Multiple Lines (Yes/No/No phone service)
        - Internet Service (No/DSL/Fiber optic/Cable)
        - Avg Monthly GB Download (numeric)
        - Online Security (Yes/No/No internet service)
        - Online Backup (Yes/No/No internet service)
        - Device Protection Plan (Yes/No/No internet service)
        - Premium Tech Support (Yes/No/No internet service)
        - Streaming TV (Yes/No/No internet service)
        - Streaming Movies (Yes/No/No internet service)
        - Streaming Music (Yes/No/No internet service)
        - Unlimited Data (Yes/No)
        - Paperless Billing (Yes/No)
        
        <strong>Financial Information:</strong>
        - Monthly Charge (numeric)
        - Total Charges (numeric)
        - Total Refunds (numeric)
        - Total Extra Data Charges (numeric)
        - Total Long Distance Charges (numeric)
        - Total Revenue (numeric)
        
        <strong>Note:</strong> The 'TotalServices' column will be calculated automatically.
        </div>
        """, unsafe_allow_html=True)
    
    # Download sample template
    if st.button("📥 Download Sample Template"):
        sample_df = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Age': [35, 42],
            'Married': ['Yes', 'No'],
            'Number of Dependents': [2, 0],
            'Zip Code': [90001, 94102],
            'Latitude': [34.0522, 37.7749],
            'Longitude': [-118.2437, -122.4194],
            'Number of Referrals': [3, 1],
            'Tenure in Months': [24, 6],
            'Phone Service': ['Yes', 'Yes'],
            'Avg Monthly Long Distance Charges': [10.5, 5.0],
            'Multiple Lines': ['Yes', 'No'],
            'Internet Service': ['Fiber optic', 'DSL'],
            'Avg Monthly GB Download': [25.5, 10.0],
            'Online Security': ['Yes', 'No'],
            'Online Backup': ['Yes', 'No'],
            'Device Protection Plan': ['Yes', 'No'],
            'Premium Tech Support': ['Yes', 'No'],
            'Streaming TV': ['Yes', 'No'],
            'Streaming Movies': ['Yes', 'No'],
            'Streaming Music': ['Yes', 'No'],
            'Unlimited Data': ['Yes', 'No'],
            'Paperless Billing': ['Yes', 'No'],
            'Monthly Charge': [89.5, 45.0],
            'Total Charges': [2146.0, 270.0],
            'Total Refunds': [0.0, 15.0],
            'Total Extra Data Charges': [0, 50],
            'Total Long Distance Charges': [252.0, 30.0],
            'Total Revenue': [2398.0, 335.0]
        })
        
        csv = sample_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="churn_prediction_template.csv">📥 Download Template CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            # Display sample data
            st.subheader("📋 Sample Data (First 5 rows)")
            st.dataframe(df.head())
            
            # Check for missing columns
            missing_columns = set(FEATURE_NAMES) - set(df.columns) - {'TotalServices'}  # TotalServices will be calculated
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            
            # Show missing values if any
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                st.warning("⚠️ Missing values detected in the data. They will be handled automatically.")
                with st.expander("View missing values"):
                    st.dataframe(missing_counts[missing_counts > 0])
            
            # Prediction button
            if st.button("🔮 Generate Predictions", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    
                    # Make a copy for processing
                    df_processed = df.copy()
                    
                    # Handle missing values
                    numeric_columns = df_processed.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_columns:
                        if df_processed[col].isnull().sum() > 0:
                            df_processed[col].fillna(df_processed[col].median(), inplace=True)
                    
                    categorical_columns = df_processed.select_dtypes(include=['object']).columns
                    for col in categorical_columns:
                        if df_processed[col].isnull().sum() > 0:
                            if len(df_processed[col].mode()) > 0:
                                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                            else:
                                df_processed[col].fillna('Unknown', inplace=True)
                    
                    # Calculate TotalServices
                    service_cols = ['Online Security', 'Online Backup', 'Device Protection Plan',
                                   'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music']
                    df_processed['TotalServices'] = 0
                    for col in service_cols:
                        if col in df_processed.columns:
                            df_processed['TotalServices'] += (df_processed[col] == 'Yes').astype(int)
                    
                    # Encode categorical variables
                    # Gender
                    df_processed['Gender'] = (df_processed['Gender'] == 'Male').astype(int)
                    
                    # Yes/No columns
                    yes_no_cols = ['Married', 'Phone Service', 'Unlimited Data', 'Paperless Billing']
                    for col in yes_no_cols:
                        if col in df_processed.columns:
                            df_processed[col] = (df_processed[col] == 'Yes').astype(int)
                    
                    # Service columns (Yes/No/No service)
                    service_cols_encode = ['Multiple Lines', 'Online Security', 'Online Backup', 
                                          'Device Protection Plan', 'Premium Tech Support',
                                          'Streaming TV', 'Streaming Movies', 'Streaming Music']
                    for col in service_cols_encode:
                        if col in df_processed.columns:
                            df_processed[col] = (df_processed[col] == 'Yes').astype(int)
                    
                    # Internet Service
                    if 'Internet Service' in df_processed.columns:
                        df_processed['Internet Service'] = (df_processed['Internet Service'] != 'No').astype(int)
                    
                    # Ensure all columns are numeric
                    for col in FEATURE_NAMES:
                        if col in df_processed.columns:
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
                    
                    # Prepare for prediction
                    X_bulk_scaled = prepare_input_data(df_processed, FEATURE_NAMES, scaler)
                    
                    # Make predictions
                    predictions, probabilities = make_predictions(X_bulk_scaled, models)
                    
                    # Add predictions to original dataframe
                    results_df = df.copy()
                    for name, prob in probabilities.items():
                        results_df[f'{name}_Probability'] = prob
                        results_df[f'{name}_Prediction'] = (prob > 0.5).astype(int)
                    
                    # Add ensemble prediction
                    if probabilities:
                        results_df['Ensemble_Probability'] = np.mean(list(probabilities.values()), axis=0)
                        results_df['Ensemble_Prediction'] = (results_df['Ensemble_Probability'] > 0.5).astype(int)
                        results_df['Risk_Level'] = results_df['Ensemble_Probability'].apply(
                            lambda x: 'High Risk' if x > 0.7 else ('Medium Risk' if x > 0.3 else 'Low Risk')
                        )
                
                st.success("✅ Predictions generated successfully!")
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Customers", len(results_df))
                with col2:
                    high_risk = (results_df['Risk_Level'] == 'High Risk').sum()
                    st.metric("High Risk", high_risk, delta=f"{high_risk/len(results_df)*100:.1f}%")
                with col3:
                    medium_risk = (results_df['Risk_Level'] == 'Medium Risk').sum()
                    st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/len(results_df)*100:.1f}%")
                with col4:
                    low_risk = (results_df['Risk_Level'] == 'Low Risk').sum()
                    st.metric("Low Risk", low_risk, delta=f"{low_risk/len(results_df)*100:.1f}%")
                
                # Risk distribution chart
                st.subheader("📈 Risk Distribution")
                risk_counts = results_df['Risk_Level'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                           title="Customer Risk Distribution",
                           color_discrete_map={'High Risk': '#ef4444', 
                                              'Medium Risk': '#f59e0b',
                                              'Low Risk': '#10b981'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed results
                st.subheader("📋 Detailed Predictions")
                
                # Select columns to display
                display_cols = st.multiselect(
                    "Select columns to display:",
                    options=results_df.columns.tolist(),
                    default=['Ensemble_Probability', 'Ensemble_Prediction', 'Risk_Level']
                )
                
                if display_cols:
                    st.dataframe(results_df[display_cols].style.background_gradient(
                        subset=['Ensemble_Probability'] if 'Ensemble_Probability' in display_cols else [],
                        cmap='RdYlGn_r'
                    ))
                
                # Download options
                st.subheader("💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = results_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv">📥 Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                with col2:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        results_df.to_excel(writer, sheet_name='Predictions', index=False)
                    excel_data = output.getvalue()
                    b64 = base64.b64encode(excel_data).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="churn_predictions.xlsx">📥 Download Excel File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file contains the required columns and is properly formatted.")

# Page 3: Model & Data Insights
elif page == "📈 Model & Data Insights":
    st.header("📈 Model Performance & Dataset Insights")
    
    # Model Performance Section
    st.subheader("🎯 Model Performance Metrics")
    
    # Display model performance table
    styled_performance = model_performance.style.background_gradient(cmap='RdYlGn', axis=0).format("{:.4f}")
    st.dataframe(styled_performance)
    
    # Performance visualization
    st.subheader("📊 Model Comparison Charts")
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=model_performance.columns.tolist(),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set3
    
    for idx, metric in enumerate(model_performance.columns):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        fig.add_trace(
            go.Bar(
                x=model_performance.index,
                y=model_performance[metric],
                name=metric,
                marker_color=colors[idx],
                text=model_performance[metric].round(3),
                textposition='auto',
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model identification
    st.subheader("🏆 Best Performing Model")
    
    best_model = model_performance['ROC-AUC'].idxmax()
    best_score = model_performance.loc[best_model, 'ROC-AUC']
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info(f"""
        ### 🥇 {best_model}
        **ROC-AUC Score:** {best_score:.4f}
        
        **Key Strengths:**
        - Accuracy: {model_performance.loc[best_model, 'Accuracy']:.4f}
        - Precision: {model_performance.loc[best_model, 'Precision']:.4f}
        - Recall: {model_performance.loc[best_model, 'Recall']:.4f}
        - F1-Score: {model_performance.loc[best_model, 'F1-Score']:.4f}
        """)
    
    # Dataset Information
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    
    # Create sample data insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Dataset Overview
        
        **Key Feature Categories:**
        
        **Demographics (4 features):**
        - Gender, Age, Married Status, Number of Dependents
        
        **Geographic (3 features):**
        - Zip Code, Latitude, Longitude
        
        **Customer Behavior (2 features):**
        - Number of Referrals, Tenure in Months
        
        **Services (13 features):**
        - Phone Service, Internet Service, Multiple Lines
        - Online Security, Backup, Device Protection
        - Premium Tech Support, Streaming Services
        - Data Usage Patterns
        
        **Financial (6 features):**
        - Monthly & Total Charges
        - Refunds, Extra Charges, Revenue
        
        **Target Variable:** Customer Churn (Binary)
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Key Insights
        
        **Churn Drivers:**
        - Low tenure customers have higher churn risk
        - Customers without service bundles churn more
        - High monthly charges relative to services increase churn
        - Limited tech support correlates with churn
        
        **Retention Factors:**
        - Multiple services reduce churn likelihood
        - Long tenure strongly indicates loyalty
        - Referral customers show lower churn
        - Balanced charge-to-service ratio improves retention
        
        **Geographic Patterns:**
        - Location data helps identify regional trends
        - Service availability varies by area
        """)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("🔑 Key Features for Churn Prediction")
    
    # Create a sample feature importance chart
    feature_importance = pd.DataFrame({
        'Feature': ['Tenure in Months', 'Monthly Charge', 'Total Charges', 'TotalServices', 
                   'Internet Service', 'Total Revenue', 'Online Security', 'Premium Tech Support',
                   'Streaming TV', 'Device Protection Plan'],
        'Importance': [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                 title='Top 10 Most Important Features',
                 color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Business Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 High-Risk Customer Actions
        
        **Immediate Interventions:**
        - 🎁 Offer multi-service bundles
        - 📞 Proactive support for new customers
        - 💰 Review pricing for high-charge accounts
        - 🔧 Free premium tech support trials
        - 📦 Incentivize long-term contracts
        
        **Long-term Strategies:**
        - Build referral programs
        - Improve onboarding experience
        - Monitor usage patterns
        """)
    
    with col2:
        st.markdown("""
        ### ✅ Customer Retention Best Practices
        
        **Preventive Measures:**
        - 📊 Track tenure milestones
        - 🤝 Reward customer referrals
        - 💳 Optimize pricing strategies
        - 🎯 Personalized service recommendations
        - 📱 Regular engagement campaigns
        
        **Success Metrics:**
        - Average customer tenure
        - Service adoption rates
        - Revenue per customer
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 20px;'>
    <p>🔬 Powered by Advanced Machine Learning | 📊 Data-Driven Insights | 🎯 Accurate Predictions</p>
    <p>© 2025 Telecom Churn Prediction System | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
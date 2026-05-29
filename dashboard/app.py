import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils import validate_numeric_value, explain_prediction


MODEL_DIR = 'models'

LANGUAGE_LABELS = {
    'app_title': {'English': 'EthioHealth-AI Clinical Assistant'},
    'app_subtitle': {'English': 'Ethiopian Clinical Decision Support with explainable AI'},
    'language_label': {'English': 'Language'},
    'sidebar_title': {'English': 'EthioHealth-AI Language Switch'},
    'page_selector': {'English': 'Navigation'},
    'page_home': {'English': 'Home'},
    'page_support': {'English': 'Support'},
    'patient_profile': {'English': 'Patient Profile'},
    'age': {'English': 'Age'},
    'gender': {'English': 'Gender'},
    'region': {'English': 'Region'},
    'fever': {'English': 'Fever'},
    'cough': {'English': 'Cough'},
    'headache': {'English': 'Headache'},
    'fatigue': {'English': 'Fatigue'},
    'vomiting': {'English': 'Vomiting'},
    'diarrhea': {'English': 'Diarrhea'},
    'chest_pain': {'English': 'Chest Pain'},
    'shortness_of_breath': {'English': 'Shortness of Breath'},
    'dizziness': {'English': 'Dizziness'},
    'temperature': {'English': 'Temperature (°C)'},
    'heart_rate': {'English': 'Heart Rate'},
    'wbc_count': {'English': 'WBC Count'},
    'hemoglobin': {'English': 'Hemoglobin'},
    'malaria_test': {'English': 'Malaria Test'},
    'comorbidity': {'English': 'Comorbidity'},
    'season': {'English': 'Season'},
    'run_assessment': {'English': 'Generate Clinical Support'},
    'predict_disease': {'English': 'Predict Disease'},
    'predict_risk': {'English': 'Predict Risk'},
    'predict_stay': {'English': 'Predict Stay'},
    'treatment_recommendation': {'English': 'Treatment Recommendation'},
    'lab_recommendation': {'English': 'Lab Test Recommendation'},
    'model_performance': {'English': 'Model Performance'},
    'ai_reason': {'English': 'Explainable AI Reason'},
    'explanation': {'English': 'Explanation'},
    'patient_summary': {'English': 'Patient Summary'},
    'confidence': {'English': 'Confidence'},
    'no_data': {'English': 'No'},
    'yes_data': {'English': 'Yes'},
    'this_estimate': {
        'English': 'This estimate is based on the patient profile and clinical risk factors.'
    },
    'overview_header': {
        'English': 'Accelerating Decisions for Ethiopian Healthcare'
    },
    'overview_text': {
        'English': 'Use EthioHealth-AI to generate fast disease, risk, and stay predictions with clear explanations. Adjust the language and patient inputs, then evaluate results in a clean clinical interface.'
    },
    'feature_disease': {
        'English': 'Disease Prediction'
    },
    'feature_risk': {
        'English': 'Risk Level Assessment'
    },
    'feature_stay': {
        'English': 'Stay Estimate'
    },
    'feature_explain': {
        'English': 'Explainable AI'
    },
    'feature_clustering': {
        'English': 'Patient Clustering'
    },
    'feature_disease_desc': {
        'English': 'Predict the most likely disease using clinical and symptom data.'
    },
    'feature_risk_desc': {
        'English': 'Estimate Low, Medium, or High risk levels for each patient.'
    },
    'feature_stay_desc': {
        'English': 'Predict how many days patients are likely to stay in the hospital.'
    },
    'feature_explain_desc': {
        'English': 'View the top features that influence each prediction.'
    },
    'feature_clustering_desc': {
        'English': 'Group patients into clusters for better clinical insights.'
    },
    'model_missing': {
        'English': 'Run `python train.py --data your_dataset.csv` to build models first.'
    },
    'home_description_title': {
        'English': 'About This Clinical Decision Support System'
    },
    'home_description': {
        'English': 'This application separates the introduction from the prediction workspace. Use Home to understand the purpose of EthioHealth-AI, then open Support to enter patient information and generate disease, risk, and hospital stay predictions.'
    },
    'support_title': {
        'English': 'Prediction Support'
    },
    'support_intro': {
        'English': 'Enter the patient profile in the sidebar, then choose the prediction you want to run.'
    },
    'description': {
        'English': 'EthioHealth-AI is designed to help clinicians by generating provisional predictions and explanations based on historical Ethiopian-style healthcare data.'
    }
}

OPTION_TRANSLATIONS = {
    'YesNo': {
        'English': {'No': 'No', 'Yes': 'Yes'}
    },
    'Gender': {
        'English': {'Male': 'Male', 'Female': 'Female', 'Other': 'Other'}
    },
    'Season': {
        'English': {'Summer': 'Summer', 'Autumn': 'Autumn', 'Winter': 'Winter', 'Spring': 'Spring'}
    },
    'PositiveNegative': {
        'English': {'Negative': 'Negative', 'Positive': 'Positive', 'Unknown': 'Unknown'}
    }
}


def get_label(key: str, lang: str) -> str:
    return LANGUAGE_LABELS.get(key, {}).get(lang, key)


def translate_option(option_type: str, selected_value: str, lang: str) -> str:
    options = OPTION_TRANSLATIONS.get(option_type, {}).get(lang, {})
    reverse_map = {display: internal for internal, display in options.items()}
    return reverse_map.get(selected_value, selected_value)


def get_option_choices(option_type: str, lang: str):
    return list(OPTION_TRANSLATIONS.get(option_type, {}).get(lang, {}).values())


def load_artifacts():
    required_files = [
        'preprocessor.joblib',
        'disease_model.joblib',
        'risk_model.joblib',
        'stay_model.joblib',
        'disease_label_encoder.joblib',
        'risk_label_encoder.joblib',
        'feature_names.joblib'
    ]
    artifacts = {}
    for filename in required_files:
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required model artifact not found: {path}")
        artifacts[filename] = joblib.load(path)
    for filename in ['recommendation_maps.joblib', 'training_metrics.joblib']:
        path = os.path.join(MODEL_DIR, filename)
        artifacts[filename] = joblib.load(path) if os.path.exists(path) else {}
    return artifacts


def build_input_dataframe(inputs):
    return pd.DataFrame([inputs])


def get_confidence(prediction_proba):
    if prediction_proba is None:
        return 0.0
    return float(np.max(prediction_proba))


def display_explanation(model, input_df, feature_names):
    try:
        explanation = explain_prediction(model, input_df, feature_names, top_n=5)
        for line in explanation:
            st.write(f"- {line}")
    except Exception as ex:
        st.warning(f"Unable to explain prediction: {ex}")


def collect_inputs(lang: str):
    st.markdown(f"#### {get_label('patient_profile', lang)}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(get_label('age', lang), min_value=0, max_value=120, value=35)
        gender_display = st.selectbox(get_label('gender', lang), get_option_choices('Gender', lang))
        region = st.selectbox(get_label('region', lang), ['Addis Ababa', 'Oromia', 'Amhara', 'Tigray', 'Southern', 'Afar', 'Somali', 'Benishangul', 'Gambela', 'Harari', 'Dire Dawa'])
        temperature = st.number_input(get_label('temperature', lang), min_value=30.0, max_value=45.0, value=37.0, format='%.1f')
        heart_rate = st.number_input(get_label('heart_rate', lang), min_value=30, max_value=200, value=80)
        wbc_count = st.number_input(get_label('wbc_count', lang), min_value=1000, max_value=50000, value=8000)
        hemoglobin = st.number_input(get_label('hemoglobin', lang), min_value=3.0, max_value=22.0, value=13.0, format='%.1f')
        weight = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=70.0, format='%.1f')
        height = st.number_input('Height (cm)', min_value=0.0, max_value=300.0, value=170.0, format='%.1f')

        
    with col2:
        bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=24.2, format='%.1f')
        oxygen_saturation = st.number_input('Oxygen Saturation (%)', min_value=0.0, max_value=100.0, value=98.0, format='%.1f')
        fever_display = st.selectbox(get_label('fever', lang), get_option_choices('YesNo', lang))
        cough_display = st.selectbox(get_label('cough', lang), get_option_choices('YesNo', lang))
        headache_display = st.selectbox(get_label('headache', lang), get_option_choices('YesNo', lang))
        fatigue_display = st.selectbox(get_label('fatigue', lang), get_option_choices('YesNo', lang))
        vomiting_display = st.selectbox(get_label('vomiting', lang), get_option_choices('YesNo', lang))
        diarrhea_display = st.selectbox(get_label('diarrhea', lang), get_option_choices('YesNo', lang))
        chest_pain_display = st.selectbox(get_label('chest_pain', lang), get_option_choices('YesNo', lang))

    with col3:
        blood_pressure_systolic = st.number_input('Blood Pressure Systolic', min_value=0, max_value=300, value=120)
        blood_pressure_diastolic = st.number_input('Blood Pressure Diastolic', min_value=0, max_value=200, value=80)
        pain_score = st.number_input('Pain Score (0-10)', min_value=0, max_value=10, value=0)
        shortness_display = st.selectbox(get_label('shortness_of_breath', lang), get_option_choices('YesNo', lang))
        dizziness_display = st.selectbox(get_label('dizziness', lang), get_option_choices('YesNo', lang))
        malaria_display = st.selectbox(get_label('malaria_test', lang), get_option_choices('PositiveNegative', lang))
        comorbidity_display = st.selectbox(get_label('comorbidity', lang), get_option_choices('YesNo', lang))
        season_display = st.selectbox(get_label('season', lang), get_option_choices('Season', lang))
        
    st.markdown("<br>", unsafe_allow_html=True)
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        symptoms = st.text_area(
            '📝 ' + 'Symptoms',
            placeholder='Enter symptoms here (e.g., high fever, coughing, chest pain)'
        )
    with scol2:
        free_text_symptoms = st.text_area(
            '📝 ' + 'Free Text Symptoms',
            placeholder='I have had a high fever for 3 days, coughing and chest pain when breathing'
        )
    with scol3:
        clinical_notes = st.text_area(
            '🩺 ' + 'Clinical Notes / Additional Info',
            placeholder='Patient has diabetes and recently traveled'
        )

    gender = translate_option('Gender', gender_display, lang)
    fever = translate_option('YesNo', fever_display, lang)
    cough = translate_option('YesNo', cough_display, lang)
    headache = translate_option('YesNo', headache_display, lang)
    fatigue = translate_option('YesNo', fatigue_display, lang)
    vomiting = translate_option('YesNo', vomiting_display, lang)
    diarrhea = translate_option('YesNo', diarrhea_display, lang)
    chest_pain = translate_option('YesNo', chest_pain_display, lang)
    shortness_of_breath = translate_option('YesNo', shortness_display, lang)
    dizziness = translate_option('YesNo', dizziness_display, lang)
    malaria_test = translate_option('PositiveNegative', malaria_display, lang)
    comorbidity = translate_option('YesNo', comorbidity_display, lang)
    season = translate_option('Season', season_display, lang)

    inputs = {
        'Age': age,
        'Gender': gender,
        'Region': region,
        'Fever': fever,
        'Cough': cough,
        'Headache': headache,
        'Fatigue': fatigue,
        'Vomiting': vomiting,
        'Diarrhea': diarrhea,
        'Chest Pain': chest_pain,
        'Shortness of Breath': shortness_of_breath,
        'Dizziness': dizziness,
        'Temperature': temperature,
        'Heart Rate': heart_rate,
        'WBC Count': wbc_count,
        'Hemoglobin': hemoglobin,
        'Malaria Test': malaria_test,
        'Comorbidity': comorbidity,
        'Season': season,
        'Weight': weight,
        'Height': height,
        'BMI': bmi,
        'Oxygen Saturation': oxygen_saturation,
        'Blood Pressure Systolic': blood_pressure_systolic,
        'Blood Pressure Diastolic': blood_pressure_diastolic,
        'Pain Score': pain_score,
        'Symptoms': symptoms,
        'Free Text Symptoms': free_text_symptoms,
        'Clinical Notes': clinical_notes
    }
    return inputs


def format_confidence(value):
    return f"{value * 100:.0f}%" if value else "N/A"


def get_recommendation(recommendation_maps, key, disease_label):
    value = recommendation_maps.get(key, {}).get(disease_label)
    if value:
        return value
    return 'Review local clinical protocol'


def build_ai_reason(inputs, disease_label):
    reasons = []
    if inputs.get('Temperature', 0) >= 38:
        reasons.append('high temperature')
    if inputs.get('Cough') == 'Yes':
        reasons.append('cough')
    if inputs.get('Chest Pain') == 'Yes':
        reasons.append('chest pain')
    if inputs.get('Shortness of Breath') == 'Yes':
        reasons.append('shortness of breath')
    if inputs.get('WBC Count', 0) >= 11000:
        reasons.append('high WBC count')
    if inputs.get('Malaria Test') == 'Positive':
        reasons.append('positive malaria test')
    if not reasons:
        reasons.append('the submitted symptoms, vitals, and lab results')
    return f"{' + '.join(reasons)} -> {disease_label}"


def render_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        :root {
            --ink: #f8fafc;
            --muted: #94a3b8;
            --line: rgba(255, 255, 255, 0.1);
            --panel: rgba(30, 41, 59, 0.7);
            --teal: #2dd4bf;
            --blue: #38bdf8;
            --amber: #fbbf24;
        }

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
        }

        .stApp {
            background: 
                radial-gradient(circle at 15% 50%, rgba(45, 212, 191, 0.15), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.15), transparent 25%),
                #0f172a;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        /* Glassmorphism Panels */
        .app-shell, .feature-card, .result-card, .status-tile, .summary-card, div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.5) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
        }

        .app-shell:hover, .feature-card:hover, .status-tile:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(45, 212, 191, 0.15);
            border-color: rgba(45, 212, 191, 0.3);
        }

        .app-shell {
            padding: 40px;
            margin-bottom: 30px;
            background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.8) 100%) !important;
            position: relative;
            overflow: hidden;
        }

        .app-shell::before {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 50%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
            transform: skewX(-20deg);
            animation: shine 6s infinite;
        }

        @keyframes shine {
            0% { left: -100%; }
            20% { left: 200%; }
            100% { left: 200%; }
        }

        .eyebrow {
            color: var(--teal);
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 12px;
            text-shadow: 0 0 10px rgba(45, 212, 191, 0.4);
        }

        .hero-title {
            color: #ffffff;
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 15px;
            background: linear-gradient(to right, #2dd4bf, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1.15rem;
            line-height: 1.6;
            max-width: 800px;
            margin: 0;
        }

        .status-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-top: 30px;
        }

        .status-tile {
            padding: 20px;
            text-align: center;
        }

        .status-value {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            background: linear-gradient(135deg, #2dd4bf, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-label {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .section-title {
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 800;
            margin: 20px 0 15px;
            position: relative;
            display: inline-block;
        }

        .section-title::after {
            content: '';
            position: absolute;
            bottom: -5px; left: 0;
            width: 40px; height: 3px;
            background: var(--teal);
            border-radius: 2px;
        }

        .section-copy {
            color: var(--muted);
            line-height: 1.7;
            margin-bottom: 24px;
            font-size: 1.05rem;
        }

        .feature-card {
            min-height: 180px;
            padding: 24px;
            border-top: 4px solid var(--teal);
        }

        .feature-code {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 12px;
            color: #ffffff;
            font-weight: 800;
            font-size: 1.2rem;
            margin-bottom: 16px;
        }

        .feature-title {
            color: #ffffff;
            font-weight: 800;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }

        .feature-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .result-card {
            padding: 24px;
            border-left: 4px solid var(--blue);
            margin-bottom: 16px;
            border-top: none;
        }

        .result-label {
            color: var(--teal);
            font-size: 0.9rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .result-value {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 8px;
        }

        .summary-card {
            padding: 24px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 16px;
        }

        .summary-item {
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 12px 16px;
            background: rgba(15, 23, 42, 0.4);
            transition: all 0.2s ease;
        }

        .summary-item:hover {
            background: rgba(30, 41, 59, 0.6);
            border-color: rgba(45, 212, 191, 0.3);
        }

        .summary-key {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .summary-value {
            color: #ffffff;
            font-weight: 800;
            font-size: 1.1rem;
            margin-top: 4px;
            word-break: break-word;
        }

        /* Modern Button Styling */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
            color: #ffffff;
            font-weight: 800;
            font-size: 1.05rem;
            min-height: 52px;
            box-shadow: 0 4px 15px rgba(13, 148, 136, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(13, 148, 136, 0.5);
            background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        }

        .stButton > button:active {
            transform: translateY(1px);
        }

        hr {
            border-color: rgba(255, 255, 255, 0.1);
            margin: 2.5rem 0;
        }

        @media (max-width: 760px) {
            .hero-title {
                font-size: 2.2rem;
            }
            .app-shell {
                padding: 30px 20px;
            }
            .status-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header(lang: str):
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="eyebrow">Clinical Decision Support</div>
            <div class="hero-title">{get_label('app_title', lang)}</div>
            <p class="hero-subtitle">{get_label('app_subtitle', lang)}</p>
            <div class="status-row">
                <div class="status-tile">
                    <div class="status-value">5</div>
                    <div class="status-label">Clinical outputs</div>
                </div>
                <div class="status-tile">
                    <div class="status-value">10+</div>
                    <div class="status-label">ML models trained</div>
                </div>
                <div class="status-tile">
                    <div class="status-value">2</div>
                    <div class="status-label">Interface languages</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_feature_card(code: str, title: str, description: str, accent: str = '#2dd4bf'):
    st.markdown(
        f"""
        <div class="feature-card" style="border-top-color: {accent};">
            <div class="feature-code" style="background: rgba(255,255,255,0.1); border: 1px solid {accent}40; color: {accent}; box-shadow: 0 0 15px {accent}30;">{code}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_result(label: str, value: str, detail: str, accent: str):
    st.markdown(
        f"""
        <div class="result-card" style="border-left-color: {accent};">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <div class="feature-copy">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_patient_summary(inputs):
    long_keys = ['Symptoms', 'Free Text Symptoms', 'Clinical Notes']
    summary_items = ''.join(
        f"""
        <div class="summary-item">
            <div class="summary-key">{key}</div>
            <div class="summary-value">{value}</div>
        </div>
        """
        for key, value in inputs.items() if key not in long_keys and str(value).strip() != ''
    )
    long_items = ''.join(
        f"""
        <div class="summary-item" style="grid-column: 1 / -1; margin-top: 8px;">
            <div class="summary-key">{key}</div>
            <div class="summary-value" style="white-space: pre-wrap; font-weight: 500; font-size: 0.95rem; line-height: 1.5;">{value}</div>
        </div>
        """
        for key, value in inputs.items() if key in long_keys and str(value).strip() != ''
    )
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-grid">
                {summary_items}
                {long_items}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title='EthioHealth-AI Clinical Assistant',
        page_icon='🩺',
        layout='wide'
    )

    render_styles()

    if 'lang' not in st.session_state:
        st.session_state.lang = 'English'

    lang = 'English'
    st.sidebar.markdown('### EthioHealth-AI')
    st.sidebar.caption(get_label('description', lang))
    
    render_header(lang)

    st.markdown(f"<h3 class='section-title'>{get_label('support_title', lang)}</h3>", unsafe_allow_html=True)
    st.write(get_label('support_intro', lang))

    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("This is likely due to a scikit-learn version mismatch or corrupt model files. Try retraining models locally with the same environment or rebooting the Streamlit Cloud server.")
        return

    inputs = collect_inputs(lang)
    input_df = build_input_dataframe(inputs)
    preprocessor = artifacts['preprocessor.joblib']

    try:
        processed_input = preprocessor.transform(input_df)
    except Exception as ex:
        st.error(f"Input preprocessing failed: {ex}")
        return

    disease_model = artifacts['disease_model.joblib']
    risk_model = artifacts['risk_model.joblib']
    stay_model = artifacts['stay_model.joblib']
    disease_encoder = artifacts['disease_label_encoder.joblib']
    risk_encoder = artifacts['risk_label_encoder.joblib']
    feature_names = artifacts['feature_names.joblib']

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(get_label('predict_disease', lang)):
            disease_pred = disease_model.predict(input_df)
            disease_proba = disease_model.predict_proba(input_df) if hasattr(disease_model, 'predict_proba') else None
            disease_label = disease_encoder.inverse_transform(disease_pred)[0]
            st.success(f"{get_label('predict_disease', lang)}: {disease_label}")
            st.write(f"{get_label('confidence', lang)}: {get_confidence(disease_proba):.2f}")
            st.markdown(f"**{get_label('explanation', lang)}**")
            display_explanation(disease_model, processed_input, feature_names)

    with col2:
        if st.button(get_label('predict_risk', lang)):
            risk_pred = risk_model.predict(input_df)
            risk_proba = risk_model.predict_proba(input_df) if hasattr(risk_model, 'predict_proba') else None
            risk_label = risk_encoder.inverse_transform(risk_pred)[0]
            st.warning(f"{get_label('predict_risk', lang)}: {risk_label}")
            st.write(f"{get_label('confidence', lang)}: {get_confidence(risk_proba):.2f}")
            st.markdown(f"**{get_label('explanation', lang)}**")
            display_explanation(risk_model, processed_input, feature_names)

    with col3:
        if st.button(get_label('predict_stay', lang)):
            stay_pred = stay_model.predict(input_df)
            st.info(f"{get_label('predict_stay', lang)}: {float(stay_pred[0]):.1f} days")
            st.write(get_label('this_estimate', lang))

    st.markdown('---')

    metrics = artifacts.get('training_metrics.joblib', {})
    disease_label_placeholder = "Pending Prediction"
    ai_reason = build_ai_reason(inputs, disease_label_placeholder)

    st.markdown(f"<div class='section-title'>{get_label('ai_reason', lang)}</div>", unsafe_allow_html=True)
    render_result(get_label('explanation', lang), ai_reason, 'Key clinical signals supporting the prediction.', '#0f766e')
    with st.expander('Feature importance details'):
        display_explanation(disease_model, processed_input, feature_names)

    st.markdown(f"<div class='section-title'>{get_label('model_performance', lang)}</div>", unsafe_allow_html=True)
    perf1, perf2, perf3 = st.columns(3)
    with perf1:
        st.metric('Disease Accuracy', f"{metrics.get('best_disease_accuracy', 0) * 100:.1f}%")
        st.caption(f"Best model: {metrics.get('best_disease_model', 'N/A')}")
    with perf2:
        st.metric('Risk Accuracy', f"{metrics.get('best_risk_accuracy', 0) * 100:.1f}%")
        st.caption(f"Best model: {metrics.get('best_risk_model', 'N/A')}")
    with perf3:
        st.metric('LOS RMSE', f"{metrics.get('stay_rmse', 0):.2f} days")
        st.caption(f"Training rows: {metrics.get('training_rows', 'N/A')}")

    st.markdown('<hr />', unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{get_label('patient_summary', lang)}</div>", unsafe_allow_html=True)
    render_patient_summary(inputs)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"A fatal error occurred during app execution: {e}")
        st.code(traceback.format_exc())
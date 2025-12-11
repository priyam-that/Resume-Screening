from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import clean_resume


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"
ROLE_DB_PATH = BASE_DIR / "role_database.csv"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run 'python train.py' first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_role_database():
    """Load the role database with descriptions, certifications, and skills."""
    if not ROLE_DB_PATH.exists():
        return None
    return pd.read_csv(ROLE_DB_PATH)


st.title("Resume Screening App")
st.write("Paste a resume below to predict its category and get detailed role information.")

resume_text = st.text_area("Resume text", height=300)

if st.button("Predict Category"):
    if not resume_text.strip():
        st.warning("Please paste a resume first.")
    else:
        model = load_model()
        role_db = load_role_database()
        cleaned = clean_resume(resume_text)
        pred = model.predict([cleaned])[0]
        
        # Match against role database
        text_low = cleaned.lower()
        
        if role_db is not None:
            # Calculate match score for each role
            role_scores = {}
            for idx, row in role_db.iterrows():
                keywords = [k.strip() for k in row['keywords'].split(',')]
                score = sum(1 for k in keywords if k in text_low)
                if score > 0:
                    role_scores[row['role_name']] = score
            
            # Override if strong match
            if role_scores:
                best_role = max(role_scores.items(), key=lambda x: x[1])
                if best_role[1] >= 3:
                    pred = best_role[0]
            
            # Display detailed info
            role_info = role_db[role_db['role_name'] == pred]
            if not role_info.empty:
                info = role_info.iloc[0]
                st.success(f"**Predicted Role:** {pred}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Experience Level", info['experience_level'])
                    st.metric("Salary Range", info['salary_range'])
                
                with col2:
                    st.write("**Description:**")
                    st.write(info['description'])
                
                st.write("**Required Skills:**")
                st.info(info['required_skills'])
                
                st.write("**Relevant Certifications:**")
                certs = info['certifications'].split(',')
                for cert in certs:
                    st.write(f"• {cert.strip()}")
            else:
                st.success(f"Predicted category: {pred}")
        else:
            st.success(f"Predicted category: {pred}")


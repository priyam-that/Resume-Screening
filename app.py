from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import clean_resume


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"
ROLE_DB_PATH = BASE_DIR / "data" / "role_database.csv"


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
st.write("Paste a resume below to get the top 3 recommended positions with detailed information.")

resume_text = st.text_area("Resume text", height=300, placeholder="Paste your resume here...")

if st.button("Analyze Resume & Get Recommendations"):
    if not resume_text.strip():
        st.warning("Please paste a resume first.")
    else:
        model = load_model()
        role_db = load_role_database()
        cleaned = clean_resume(resume_text)
        pred = model.predict([cleaned])[0]
        
        # Match against role database
        text_low = cleaned.lower()
        
        top_matches = []
        
        if role_db is not None:
            # Calculate match score for each role
            role_scores = {}
            for idx, row in role_db.iterrows():
                keywords = [k.strip() for k in row['keywords'].split(',')]
                score = sum(1 for k in keywords if k in text_low)
                if score > 0:
                    role_scores[row['role_name']] = score
            
            # Get top 3 matches
            if role_scores:
                sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
                top_matches = sorted_roles[:3]
            
            # Display top 3 recommendations
            if top_matches:
                st.success(f"✅ Analysis Complete! Here are your top {len(top_matches)} recommended positions:")
                
                for rank, (role_name, score) in enumerate(top_matches, 1):
                    role_info = role_db[role_db['role_name'] == role_name]
                    if not role_info.empty:
                        info = role_info.iloc[0]
                        
                        with st.expander(f"#{rank} - {role_name} (Match Score: {score})", expanded=(rank==1)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Experience Level", info['experience_level'].title())
                                st.metric("Salary Range", f"${info['salary_range']}")
                                st.metric("Match Score", f"{score} keywords")
                            
                            with col2:
                                st.write("**📋 Description:**")
                                st.info(info['description'])
                            
                            st.write("**🛠️ Required Skills:**")
                            skills = info['required_skills'].split(',')
                            st.write(", ".join([f"`{s.strip()}`" for s in skills]))
                            
                            st.write("**📜 Relevant Certifications:**")
                            certs = info['certifications'].split(',')
                            for cert in certs:
                                st.write(f"• {cert.strip()}")
                
                # Summary recommendation
                st.markdown("---")
                st.markdown("### 💡 Recommendation")
                best_match = top_matches[0][0]
                st.write(f"Based on your resume, **{best_match}** is the strongest match for your skills and experience.")
                
            else:
                st.success(f"Predicted category: {pred}")
        else:
            st.success(f"Predicted category: {pred}")


# 🎯 AI-Powered Resume Screening System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange.svg" alt="ML">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

An intelligent resume screening system that automatically categorizes resumes into 18+ specific technical roles using machine learning and rule-based classification. Features include a CLI tool, web interface, and comprehensive role database with certifications and salary information.

## ✨ Features

### 🤖 **Smart Classification**
- **18 specialized tech roles** including Frontend, Backend, Full Stack, DevOps, ML Engineer, Data Scientist, Cloud Architect, and more
- **Hybrid approach**: Combines ML model (TF-IDF + Logistic Regression) with keyword-based rules
- **High accuracy** with balanced class weights and role-specific keyword matching

### 📊 **Rich Role Database**
- Detailed role information stored in `role_database.csv`
- Each role includes:
  - 🔑 Keywords for matching
  - 📜 Relevant certifications (AWS, Google Cloud, CISSP, etc.)
  - 💼 Experience levels (junior/mid/senior)
  - 💰 Salary ranges
  - 🛠️ Required skills
  - 📝 Role descriptions

### 🖥️ **Multiple Interfaces**
- **CLI Tool**: Quick predictions from terminal with verbose mode
- **Web App**: Interactive Streamlit interface with visual metrics
- **Python API**: Import and use in your own scripts

### 🔧 **Extensible**
- Easily add new roles by editing CSV
- Support for multiple data formats (CSV, JSON, text files)
- Modular architecture for custom extensions

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anukalp-mishra/Resume-Screening.git
cd Resume-Screening

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

This will:
- Load `resume_dataset.csv`
- Clean and preprocess text
- Train a TF-IDF + Logistic Regression pipeline
- Evaluate with classification report
- Save model to `models/resume_classifier.joblib`

---

## 💻 Usage

### 1️⃣ Command Line Interface (Recommended)

**Basic prediction:**
```bash
python predict_cli.py "5 years Python experience, Django, REST APIs, PostgreSQL"
# Output: Backend Developer
```

**Verbose mode (full details):**
```bash
python predict_cli.py -v "React, TypeScript, Next.js, 4 years frontend experience"
```

**Output:**
```
Predicted Role: Frontend Developer
Description: Builds user-facing web applications with modern frameworks and responsive design
Experience Level: junior-senior
Salary Range: 60k-150k
Required Skills: JavaScript/TypeScript, React/Vue/Angular, HTML5/CSS3, REST APIs, Git
Relevant Certifications: AWS Certified Developer, Google Mobile Web Specialist, Meta Front-End Developer
```

**Read from file:**
```bash
cat resume.txt | python predict_cli.py -v
```

### 2️⃣ Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser and:
- Paste resume text
- Click "Predict Category"
- View predicted role with metrics, skills, and certifications

### 3️⃣ Python API

```python
from predict_cli import load_model, load_role_database
from preprocessing import clean_resume

# Load model and database
model = load_model()
role_db = load_role_database()

# Predict
resume_text = "Your resume text here..."
cleaned = clean_resume(resume_text)
prediction = model.predict([cleaned])[0]
print(f"Predicted Role: {prediction}")
```

---

## 🎯 Supported Roles

| Role | Key Technologies | Certifications |
|------|-----------------|----------------|
| **Frontend Developer** | React, Vue, Angular, TypeScript, Next.js | AWS Certified Developer, Meta Front-End |
| **Backend Developer** | Node.js, Django, Flask, FastAPI, GraphQL | AWS Developer, Azure Developer |
| **Full Stack Developer** | MERN/MEAN stack, React + Node.js | AWS Solutions Architect |
| **Mobile Developer** | React Native, Flutter, Swift, Kotlin | Google Android, Apple iOS Developer |
| **Machine Learning Engineer** | TensorFlow, PyTorch, MLOps | AWS ML Specialty, TensorFlow Developer |
| **Data Scientist** | Python, pandas, Statistics, Tableau | Google Data Analytics, IBM Data Science |
| **Data Engineer** | Spark, Airflow, ETL, Snowflake | AWS Data Analytics, Databricks |
| **DevOps Engineer** | Kubernetes, Docker, CI/CD, Terraform | AWS DevOps, CKA |
| **Cloud Architect** | AWS/Azure/GCP, Multi-cloud | Solutions Architect Professional |
| **Site Reliability Engineer** | Monitoring, Incident Response, SRE | AWS SysOps, CKA |
| **Security Engineer** | Penetration Testing, OWASP | CISSP, CEH, Security+ |
| **Blockchain Developer** | Solidity, Web3, Smart Contracts | Certified Blockchain Developer |
| **QA Engineer** | Selenium, Cypress, Test Automation | ISTQB, Selenium Professional |
| **Product Manager** | Product Strategy, Agile, Roadmaps | CSPO, Product Management Cert |
| **UI/UX Designer** | Figma, User Research, Prototyping | Google UX, Adobe Certified Expert |
| **Database Administrator** | MySQL, PostgreSQL, Query Tuning | Oracle DBA, Azure Database Admin |
| **Embedded Systems Engineer** | Firmware, IoT, RTOS, C/C++ | Embedded Systems Engineer Cert |
| **Game Developer** | Unity, Unreal Engine, C# | Unity Certified Developer |

---

## 📁 Project Structure

```
Resume-Screening/
├── app.py                      # Streamlit web application
├── predict_cli.py              # Command-line prediction tool
├── train.py                    # Model training script
├── preprocessing.py            # Text cleaning functions
├── data_loader.py              # Multi-format data loader
├── role_database.csv           # Role information database
├── resume_dataset.csv          # Training data
├── requirements.txt            # Python dependencies
├── models/
│   └── resume_classifier.joblib  # Trained model
├── data/                       # Additional training data (CSV/JSON/text)
├── Resume_Screening.ipynb      # Jupyter notebook for exploration
├── README.md                   # This file
└── ROLE_DATABASE_GUIDE.md      # Guide for adding custom roles
```

---

## 🔧 Customization

### Adding New Roles

Edit `role_database.csv` to add new roles:

```csv
role_name,keywords,certifications,experience_level,description,salary_range,required_skills
Technical Writer,"documentation,technical writing,api docs,markdown","Technical Writing Certification",junior-senior,"Creates technical documentation and API guides","50k-100k","Writing, Markdown, APIs, Git"
```

See [ROLE_DATABASE_GUIDE.md](ROLE_DATABASE_GUIDE.md) for detailed instructions.

### Adding Training Data

Add new data sources to the `data/` directory:

- **CSV files**: `data/new_resumes.csv` (columns: `text`, `category`)
- **JSON files**: `data/resumes.json` (array of objects with `text` and `category`)
- **Text files**: Organize by category: `data/text_resumes/Frontend Developer/*.txt`

Then retrain:
```bash
python train.py
```

---

## 🧪 Example Predictions

```bash
# Frontend Developer
python predict_cli.py "HTML, CSS, JavaScript, React, TypeScript, responsive design"
# Output: Frontend Developer

# Data Engineer
python predict_cli.py "ETL pipelines, Apache Spark, Airflow, data warehousing, SQL"
# Output: Data Engineer

# DevOps Engineer
python predict_cli.py "Kubernetes, Docker, CI/CD, Jenkins, Terraform, AWS"
# Output: DevOps Engineer

# Full Stack Developer
python predict_cli.py "React frontend, Node.js backend, MongoDB, REST APIs"
# Output: Full Stack Developer
```

---

## 📊 Model Performance

The classifier uses:
- **TF-IDF Vectorization** with 5000 features
- **Logistic Regression** with balanced class weights
- **Stratified train-test split** (80/20)
- **Rule-based override** for high-confidence keyword matches (3+ keywords)

Typical accuracy: **85-92%** depending on training data quality.

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **ML/AI** | scikit-learn, NLTK, pandas, numpy |
| **Web** | Streamlit |
| **Data Processing** | pandas, regex, joblib |
| **Visualization** | matplotlib, seaborn, wordcloud |
| **Development** | Python 3.8+, Jupyter |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

**Ideas for contributions:**
- Add more roles to the database
- Improve keyword matching algorithms
- Add support for PDF/DOCX parsing
- Implement experience level detection
- Add multi-language support

---

## 📝 Why Resume Screening Automation?

### The Problem
- Companies receive **thousands of resumes** for each job posting
- Manual screening is **time-consuming** and **inconsistent**
- Hiring teams struggle to **identify qualified candidates** quickly
- **Days of work** can be done in **minutes** with automation

### The Solution
This system uses:
- **Machine Learning** to learn patterns from existing resumes
- **Natural Language Processing** to extract relevant information
- **Rule-based classification** for domain-specific accuracy
- **Comprehensive role database** for detailed insights

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Anukalp Mishra**

- GitHub: [@anukalp-mishra](https://github.com/anukalp-mishra)
- Repository: [Resume-Screening](https://github.com/anukalp-mishra/Resume-Screening)

---

## 🙏 Acknowledgments

- Original dataset from various resume sources
- scikit-learn and NLTK communities
- Streamlit for the amazing web framework
- Open-source contributors

---

## 📞 Support

If you encounter issues or have questions:

1. Check [ROLE_DATABASE_GUIDE.md](ROLE_DATABASE_GUIDE.md) for customization help
2. Open an issue on GitHub
3. Review existing issues for similar problems

---

<p align="center">
  <strong>Made with ❤️ for smarter recruitment</strong>
</p>

<p align="center">
  ⭐ Star this repo if you find it helpful!
</p>

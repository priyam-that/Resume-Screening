# Resume Screening

This repository contains a machine learning project focused on automating the resume screening process using Python. The primary goal is to develop a model that can efficiently and accurately evaluate resumes based on predefined criteria.

<img src="Cover.png" alt="resume cover">

## Table of Contents

- [Why do we need Resume Screening?](#why-do-we-need-resume-screening)
- [Introduction](#introduction)
- [Modules & Libraries](#modules--libraries)
- [Functionality of Application](#functionality-of-application)
- [Tools & Technologies Used](#tools--technologies-used)
- [Tech Innovations in Resume Screening](#tech-innovations-in-resume-screening)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Why do we need Resume Screening?

- For each recruitment, companies take out the resume, referrals and go through them manually.
- Companies often received thousands of resumes for every job posting.
- When companies collect resumes then they categorize those resumes according to their requirements and then they send the collected resumes to the Hiring Teams.
- It becomes very difficult for the hiring teams to read the resume and select the resume according to the requirement, there is no problem if there are one or two resumes but it is very difficult in case of hundreds of resumes.
- To solve this problem, we will screen the resume using machine learning and NLP using Python so that we can complete days of work in few minutes.

## Introduction

- Resume screening is the process of determining whether a candidate is qualified for a role based on their education, experience, and other information captured on their resume.
- It’s a form of pattern matching between a job’s requirements and the qualifications of a candidate based on their resume.
- The goal of screening resumes is to decide whether to move a candidate forward – usually onto an interview – or to reject them.

## Modules & Libraries

### Modules
- **KNN**: It's a supervised technique used for classification. "K" in KNN represents the number of nearest neighbors used to classify or predict in case of continuous variables.
- **NLP**: Natural Language Processing (NLP) is a field in machine learning with the ability of a computer to understand, analyze, manipulate, and potentially generate human language.

### Libraries
- **NumPy**: Fundamental package for Python providing support for large multidimensional arrays and matrices.
- **Pandas**: Open-source library providing easy data structures and quicker data analysis for Python.
- **Matplotlib**: Open-source library widely used for creating publication-quality figures in a variety of formats.
- **Seaborn**: Library derived from Matplotlib used for visualizing statistical models like heat maps.
- **SciPy**: Open-source software used for scientific computing in Python.
- **Scikit-learn**: Free software machine learning library for Python used for classification, regression, clustering, and more.
- **NLTK**: Natural Language Toolkit (NLTK) is a set of processing libraries providing solutions for text analysis and language processing.

## Functionality of Application

Screening resumes usually involves a three-step process based on the role’s minimum and preferred qualifications. Both types of qualifications should be related to on-the-job performance. These qualifications can include:
- Work experience
- Education
- Skills and knowledge
- Personality traits
- Competencies

## Tools & Technologies Used

- Machine Learning and Artificial intelligence, along with text mining and natural language processing algorithms, can be applied for the development of programs (i.e. Applicant Tracking Systems) to automate the resume screening process.

## Tech Innovations in Resume Screening

- Designed to meet the needs of recruiters that current technology can’t solve, a new class of recruiting technology called AI for recruitment has arrived.
- AI for recruiting is an emerging category of HR technology designed to reduce — or even remove — time-consuming, administrative activities like manually screening resumes.
- The best AI software is designed to integrate seamlessly with your current recruiting stack so it doesn’t disrupt your workflow nor the candidate workflow.
- Industry experts predict this type of automation technology will transform the recruiting function.

## Installation

To get started with the upgraded project, clone the repository, create a virtual environment, and install the dependencies.

```bash
git clone https://github.com/anukalp-mishra/Resume-Screening.git
cd Resume-Screening

# create and activate a virtual environment (example)
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Train the model (script)

Run the standalone training script, which loads `resume_dataset.csv`, cleans the text, trains a TF‑IDF + Logistic Regression pipeline, evaluates it, and saves the model to `models/resume_classifier.joblib`.

```bash
python train.py
```

### 2. Predict from the command line

After training, you can predict a resume category directly from the terminal using `predict_cli.py`:

```bash
# Basic prediction (just shows the role name)
python predict_cli.py "Paste resume text here..."

# Verbose mode (shows description, salary, skills, certifications)
python predict_cli.py --verbose "5 years React, TypeScript, Next.js experience..."
python predict_cli.py -v "ETL pipelines, Spark, Airflow, data engineering..."

# or read from stdin
cat my_resume.txt | python predict_cli.py
cat my_resume.txt | python predict_cli.py --verbose
```

The CLI uses a **role database** (`role_database.csv`) that contains 18 specific roles with:
- Keywords for matching
- Role descriptions
- Experience levels (junior/mid/senior)
- Salary ranges
- Required skills
- Relevant certifications (AWS, Google Cloud, CISSP, etc.)

You can edit `role_database.csv` to add more roles, update keywords, or modify descriptions.

### 3. Run the Streamlit web app

Launch a simple web UI for interactive resume screening:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) and paste a resume to see the predicted category.

### 4. Explore the notebook

The original Jupyter notebook `Resume_Screening.ipynb` is still available for exploration, EDA, and visualization.

```bash
jupyter notebook Resume_Screening.ipynb
```

## Dataset
The dataset used for this project consists of resumes collected from various sources. The data is preprocessed to extract relevant features such as skills, experience, and education.

## Model
The project uses various machine learning models to evaluate resumes. These models are trained on labeled data to classify resumes based on predefined criteria. The models include:

Logistic Regression
Support Vector Machines
Random Forest
Neural Networks

## Results
The performance of the models is evaluated using metrics such as accuracy, precision, recall, and F1-score. The results are documented and visualized in the Jupyter notebooks.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.


## License
This project is licensed under the MIT License. See the LICENSE file for more details.


## Contact
For any questions or inquiries, please contact:

Anukalp Mishra

GitHub: anukalp-mishra

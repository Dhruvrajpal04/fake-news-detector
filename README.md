# FakeNewsLens: Machine Learning Fake News Detection System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-000000.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end, production-grade, college-level Machine Learning Web Application designed to analyze news headlines and articles, extract stylistic and lexical patterns using TF-IDF (unigrams & bigrams), and predict whether the text resembles patterns labeled **REAL** or **FAKE** in the peer-reviewed **WELFake** benchmark dataset.

---

## Table of Contents
1. [Important Ethical & Responsible AI Notice](#important-ethical--responsible-ai-notice)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Dataset Provenance & Citation](#dataset-provenance--citation)
5. [Project Architecture & Directory Layout](#project-architecture--directory-layout)
6. [Windows Setup & Quick Start Guide](#windows-setup--quick-start-guide)
7. [Training the Machine Learning Pipeline](#training-the-machine-learning-pipeline)
8. [Running Automated Tests](#running-automated-tests)
9. [Web Application Walkthrough](#web-application-walkthrough)
10. [College Viva & Technical Defense Guide](#college-viva--technical-defense-guide)
11. [Independent Fact-Checking References](#independent-fact-checking-references)
12. [Troubleshooting Common Issues](#troubleshooting-common-issues)
13. [Future Roadmap](#future-roadmap)

---

## Important Ethical & Responsible AI Notice

> **Critical Disclaimer:**
> Machine learning text classification evaluates **linguistic syntax, n-gram frequencies, and rhetorical style** learned from training data.
> **A prediction of "Likely Real" or "Likely Fake" does NOT prove factual truth or falsehood in the physical world.**
> 
> - A truthful event described with emotional, colloquial, or eccentric phrasing may trigger a "Likely Fake" classification.
> - A fabricated conspiracy written in polished, formal journalistic prose may trigger a "Likely Real" classification.
> - This application **does NOT connect to live external news wires or claim to have verified facts.** Users are guided to consult independent fact-checkers.

---

## Key Features

- **Genuine Machine Learning:** Powered by a Scikit-Learn `Pipeline` combining `TfidfVectorizer` (sublinear TF, n-gram range 1–2, 50,000 maximum features) and regularized `LogisticRegression`.
- **Zero Train-Test Leakage:** Vectorizer vocabulary and IDF weights are computed strictly on the training fold (`X_train`), while evaluation is reported on holdout test samples (`X_test`).
- **Interactive Web Interface:** Modern, responsive UI adhering to a professional navy-blue (`#0f2744`), white, and light-blue (`#e0f2fe`) theme with instant sample loaders (Real News, Sensational Fake, Clickbait).
- **Comprehensive Analytics Dashboard:** Real-time Chart.js visualizers for per-class precision/recall/F1, confusion matrix, top discriminative n-grams, and live query distribution.
- **Auditable SQLite Database:** Saves evaluated article previews, timestamps, model confidence scores, and optional source URLs using safe parameterized SQL queries. Includes one-click individual deletion and complete history purge.
- **Robust Security & Input Validation:** Rejects empty submissions, too-short snippets (<15 characters), and excessively long payloads (>50,000 characters). All HTML output is sanitized to prevent XSS.
- **Extensive Test Coverage:** Automated test suite (`pytest`) covering the Flask routes, database schema/CRUD, and ML serialization/inference.

---

## Technology Stack

| Layer | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Responsive UI, mobile navigation drawer, live counters, modals |
| **Data Viz** | Chart.js 4.4 | Bar charts, doughnut charts, and feature importance visualizers |
| **Backend** | Python 3.11+, Flask 3.1+ | RESTful API, server-rendered Jinja2 templates, error handlers |
| **Machine Learning**| Scikit-learn, NumPy, SciPy | TF-IDF Vectorizer + Logistic Regression classification pipeline |
| **Data Processing** | Pandas | Handling null values, deduplication, and class auditing |
| **Persistence** | Joblib & SQLite3 | Compressed pipeline model storage & local audit database |
| **Testing** | Pytest 9.0+ | Automated unit and integration testing |

---

## Dataset Provenance & Citation

The model trains on the **WELFake** dataset:
- **Paper:** Verma, P. K., Agrawal, P., Amorim, I., & Prodan, R. (2021). *"WELFake: Word Embedding Over Linguistic Features for Fake News Detection."* IEEE Transactions on Computational Social Systems. DOI: [10.1109/TCSS.2021.3068519](https://doi.org/10.1109/TCSS.2021.3068519).
- **Zenodo Repository:** [Record 4561253](https://zenodo.org/records/4561253)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Size:** 72,134 news articles (35,028 Real, 37,106 Fake)
- **Composition:** Merged from 4 verified datasets (Reuters, BuzzFeed Political, McIntire, Kaggle) to prevent single-source classifier overfitting.
- **Binary Labels:** `0 = Fake News`, `1 = Real News`.

---

## Project Architecture & Directory Layout

```
fake-news/
├── app.py                     # Flask web server, page routes, and API endpoints
├── database.py                # SQLite schema, parameterized queries, and analytics helpers
├── download_dataset.py        # Automated Zenodo dataset downloader with chunk progress
├── train_model.py             # Preprocessing, train/test split, model training, evaluation
├── requirements.txt           # Python dependency specifications
├── pytest.ini                 # Pytest path and runner configuration
├── README.md                  # Complete documentation and viva guide
├── data/                      # Dataset and SQLite database storage
│   ├── WELFake_Dataset.csv   # Downloaded official dataset (245 MB)
│   └── predictions.db        # SQLite database (auto-created on initial launch)
├── models/                    # Model artifacts
│   ├── fake_news_pipeline.joblib  # Trained Scikit-Learn pipeline
│   └── model_metrics.json         # Evaluation metrics, confusion matrix, top n-grams
├── static/
│   ├── css/
│   │   └── style.css          # Navy & light-blue responsive theme
│   └── js/
│       └── main.js            # Form handling, AJAX, live counters, modals
├── templates/
│   ├── base.html              # Core layout, navbar, fact-checker links, footer
│   ├── index.html             # News submission form, real-time results, confidence meter
│   ├── dashboard.html         # Chart.js metrics, confusion matrix, feature charts
│   ├── history.html           # Auditable history log, full-text viewer modal, clear button
│   └── about.html             # Methodology, mathematical equations, viva questions
└── tests/
    ├── test_app.py            # Flask endpoints, validation, error handling tests
    ├── test_database.py       # SQLite CRUD, parameterization, and analytics tests
    └── test_model.py          # Pipeline construction, probabilities, and serialization tests
```

---

## Windows Setup & Quick Start Guide

### Step 1: Clone or Open the Workspace
Open PowerShell or Command Prompt in the project folder:
```powershell
cd "c:\Users\dhruv\Downloads\fake news"
```

### Step 2: Create a Python Virtual Environment
We recommend Python 3.11:
```powershell
py -3.11 -m venv venv
```

### Step 3: Activate the Virtual Environment
- **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then re-run the activate script).*
- **In Command Prompt (cmd.exe):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Training the Machine Learning Pipeline

### Option A: Automatic Download & Training (Recommended)
You can run the standalone downloader or train directly:
```powershell
python download_dataset.py
python train_model.py
```
*(The training script handles missing values, eliminates duplicate content records, fits the TF-IDF and Logistic Regression pipeline, prints precision/recall/F1 metrics, and saves the compressed model to `models/fake_news_pipeline.joblib`).*

### Option B: Quick Prototyping (Subsampled)
If you are testing quickly or in a time-constrained demo:
```powershell
python train_model.py --max-samples 10000
```

### Option C: Manual Download
1. Visit the Zenodo repository: [https://zenodo.org/records/4561253](https://zenodo.org/records/4561253)
2. Download `WELFake_Dataset.csv` (245 MB).
3. Place the file inside the `data/` directory.
4. Run `python train_model.py`.

---

## Running Automated Tests

Run the full pytest suite:
```powershell
pytest -v
```
All 18 automated tests cover:
- Database initialization, parameterized queries, and cascading clear
- Pipeline vectorization, probability checks, and Joblib serialization
- Input boundaries (rejection of empty, short, or oversized inputs)
- Web routes (`/`, `/dashboard`, `/history`, `/about`, `/api/predict`, `/api/history`)

---

## Web Application Walkthrough

### 1. Starting the Server
```powershell
python app.py
```
Output:
```
=======================================================
Fake News Detection System running at http://127.0.0.1:5000
Model status: READY
=======================================================
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### 2. Available Web Pages
- **Detect News (`/`):** Paste news text or use the quick sample loaders (Real Journalistic Report, Sensational Claim, Conspiracy Rumor). Click **Analyze News** to see real-time classification, confidence meter, stylistic explanation, and fact-checker links.
- **Dashboard (`/dashboard`):** Real-time Chart.js visualizations showing model test accuracy, per-class F1-scores, confusion matrix breakdown, and the top weighted vocabulary features.
- **History (`/history`):** Complete audit log of analyzed articles. Click "View Full Text" to review the submission or "Clear All History" to purge records.
- **About & Methodology (`/about`):** Detailed academic explanation, mathematical formulas, dataset licensing, and viva questions.

---

## College Viva & Technical Defense Guide

### Q1: What is TF-IDF and why use it over Bag-of-Words (CountVectorizer)?
**Answer:** Bag-of-Words merely counts word frequencies. This causes common words (e.g., "said", "announced", "reported") to dominate the vector representation without conveying discriminative information. TF-IDF balances Term Frequency with Inverse Document Frequency, penalizing terms that appear across almost all documents and emphasizing distinctive terms that define stylistic classes.

### Q2: How does Logistic Regression classify text?
**Answer:** Given the sparse TF-IDF feature vector $\mathbf{x} \in \mathbb{R}^d$, Logistic Regression computes a linear combination $z = \mathbf{w}^T \mathbf{x} + b$ using learned weights $\mathbf{w}$. It then applies the standard logistic sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to map the unbounded log-odds into a bounded posterior probability $P(\text{Real}|\mathbf{x}) \in [0, 1]$. If $P(\text{Real}|\mathbf{x}) \ge 0.50$, it classifies the sample as Likely Real; otherwise, Likely Fake.

### Q3: How do you prevent data leakage in your ML pipeline?
**Answer:** Data leakage occurs when test set statistics influence feature extraction. To prevent this, `train_test_split` is executed first. The `TfidfVectorizer` is fitted strictly on `X_train`. The test fold `X_test` is only transformed using the already-fitted vocabulary and IDF weights, ensuring that test evaluation reflects performance on genuinely unseen data.

### Q4: What does the confidence percentage represent?
**Answer:** The confidence percentage is the calibrated posterior probability $\max(P(\text{Fake}), P(\text{Real})) \times 100\%$. It reflects the statistical certainty with which the model maps the text's n-grams to patterns observed in the WELFake training corpus. It is **not** a guarantee of factual truth.

### Q5: Why is this approach lightweight and suitable for production?
**Answer:** Unlike heavy Transformer models (e.g. BERT or GPT) that require multi-gigabyte GPU memory, TF-IDF + Logistic Regression runs inference in less than 2 milliseconds on a commodity CPU, occupies under 30 MB of RAM, and offers full model interpretability by inspecting logistic regression coefficients.

---

## Independent Fact-Checking References

When evaluating news claims, users are directed to consult verified independent media desks:
1. **Reuters Fact Check:** [https://www.reuters.com/fact-check/](https://www.reuters.com/fact-check/) — Global investigations covering political, financial, and viral claims.
2. **Associated Press (AP) Verify:** [https://www.ap.org/verify](https://www.ap.org/verify) — In-depth reporting addressing viral misinformation.
3. **PIB Fact Check:** [https://factcheck.pib.gov.in/](https://factcheck.pib.gov.in/) — Official verification wing of the Press Information Bureau, Government of India.
4. **Newschecker India:** [https://www.newschecker.in/](https://www.newschecker.in/) — IFCN-certified fact-checking organization focusing on regional and viral media claims.

*(Note: These links are provided for manual human verification; the system does not claim an automated fact API connection).*

---

## Troubleshooting Common Issues

1. **PowerShell Script Execution Error:**
   - *Error:* `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
   - *Solution:* Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell, then re-run `.\venv\Scripts\Activate.ps1`.
2. **Model Not Loaded Error:**
   - *Error:* UI displays "Model Artifact Not Detected".
   - *Solution:* Run `python train_model.py` to create `models/fake_news_pipeline.joblib`.
3. **Port 5000 Already in Use:**
   - *Solution:* Specify a different port: `$env:PORT="5050"; python app.py`.

---

## Future Roadmap

1. **Named Entity Disambiguation (NER):** Extract specific politicians, corporations, and locations using spaCy to evaluate entity context.
2. **Cross-Lingual Classification:** Expand training to multilingual Hindi/regional news corpora.
3. **Source Domain Credibility Scoring:** Cross-reference user-supplied source URLs against established Media Bias / Fact Check (MBFC) domain credibility databases.

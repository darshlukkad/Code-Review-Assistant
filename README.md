# AI-Powered Code Review Assistant

## 🎯 Project Overview

An intelligent code review system powered by deep learning that automatically detects bugs, code smells, security vulnerabilities, and suggests improvements. Built as an end-to-end full-stack application for the CS 5590 AI/ML Final Project.

### Team Members
- Member 1: Data Engineering & Preprocessing
- Member 2: Model Development & Training
- Member 3: Full-Stack Application Development
- Member 4: Deployment & Documentation

### Key Features
✅ Multi-label code issue detection (bugs, security, code smells, style)  
✅ Support for Python and JavaScript  
✅ Real-time inference with <2s response time  
✅ Confidence scores and explainability  
✅ Production-ready web application  
✅ Comprehensive evaluation with TensorBoard  

---

## 📊 Project Methodology: CRISP-DM

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

1. **Business Understanding:** Define code quality objectives and success criteria
2. **Data Understanding:** Explore CodeSearchNet and GitHub quality datasets
3. **Data Preparation:** Preprocessing, augmentation, and dataset splitting
4. **Modeling:** Fine-tune CodeBERT and compare architectures
5. **Evaluation:** Comprehensive metrics, ablation studies, visualization
6. **Deployment:** Full-stack web app with Docker and cloud deployment

See [CRISP-DM.md](docs/CRISP-DM.md) for detailed methodology documentation.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or Google Colab
- Docker (for deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/code-review-assistant.git
cd code-review-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (if available)
python src/deployment/download_model.py
```

### Running the Application

#### Option 1: Streamlit App (Recommended)
```bash
cd app/frontend
streamlit run app.py
```
Access at: http://localhost:8501

#### Option 2: Full Stack with FastAPI Backend
```bash
# Terminal 1: Start backend
cd app/backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start frontend
cd app/frontend
streamlit run app.py
```

#### Option 3: Docker Deployment
```bash
docker-compose up --build
```
Access at: http://localhost:8501

---

## 📚 Project Structure

```
code-review-assistant/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01-EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02-preprocessing.ipynb        # Data preprocessing
│   ├── 03-model-training.ipynb       # Model training & tuning
│   └── 04-evaluation.ipynb           # Evaluation & visualization
│
├── src/                               # Source code
│   ├── data/
│   │   ├── data_loader.py            # Dataset loading utilities
│   │   └── preprocessing.py          # Preprocessing functions
│   ├── models/
│   │   └── model.py                  # Model architectures
│   ├── training/
│   │   ├── trainer.py                # Training loop
│   │   └── config.py                 # Hyperparameter config
│   ├── evaluation/
│   │   ├── evaluator.py              # Evaluation metrics
│   │   └── visualizations.py         # Plotting utilities
│   └── deployment/
│       └── inference.py              # Model inference
│
├── app/                               # Full-stack application
│   ├── backend/
│   │   ├── main.py                   # FastAPI server
│   │   └── requirements.txt          # Backend dependencies
│   └── frontend/
│       ├── app.py                    # Streamlit UI
│       └── requirements.txt          # Frontend dependencies
│
├── deployment/                        # Deployment configs
│   ├── Dockerfile                    # Docker container
│   ├── docker-compose.yml            # Multi-container setup
│   ├── kubernetes/                   # K8s manifests
│   └── cloud/                        # Cloud deployment scripts
│
├── docs/                              # Documentation
│   ├── CRISP-DM.md                   # Methodology details
│   ├── METHODOLOGY.md                # Process documentation
│   └── EVALUATION.md                 # Metrics & results
│
├── presentation/                      # Final deliverables
│   ├── slides.pptx                   # Presentation deck
│   └── demo-video.mp4                # 5-15 min demo video
│
└── tests/                             # Test suite
    ├── test_data_loader.py
    ├── test_model.py
    └── test_api.py
```

---

## 🔬 Model Architecture

### Base Model: CodeBERT
- Pre-trained on 6 programming languages
- 12 transformer layers, 768 hidden dimensions
- Fine-tuned for multi-label classification

### Training Details
- **Loss Function:** Binary Cross-Entropy (multi-label)
- **Optimizer:** AdamW (lr=2e-5, weight_decay=0.01)
- **Activation:** GELU (hidden), Sigmoid (output)
- **Batch Size:** 32
- **Epochs:** 10-15 with early stopping

See [notebooks/03-model-training.ipynb](notebooks/03-model-training.ipynb) for full implementation.

---

## 📈 Evaluation Metrics

### Model Performance
- **F1-Score:** Target ≥0.85 (macro-averaged)
- **Precision/Recall:** Per-class metrics
- **AUC-ROC:** Area under ROC curve
- **Inference Time:** <2 seconds per file

### Visualizations (20% of Project)
- Training/validation loss curves
- Confusion matrices per issue type
- ROC and Precision-Recall curves
- Ablation study comparisons
- TensorBoard dashboards

See [notebooks/04-evaluation.ipynb](notebooks/04-evaluation.ipynb) and [docs/EVALUATION.md](docs/EVALUATION.md).

---

## 🧪 Ablation Studies

| Experiment | Description | F1-Score | Inference Time |
|------------|-------------|----------|----------------|
| Baseline | CodeBERT default | TBD | TBD |
| No Augmentation | Without data augmentation | TBD | TBD |
| GraphCodeBERT | Alternative architecture | TBD | TBD |
| DistilCodeBERT | Smaller, faster model | TBD | TBD |
| Focal Loss | Different loss function | TBD | TBD |

---

## 🌐 Deployment

### Local Development
```bash
streamlit run app/frontend/app.py
```

### Docker
```bash
docker-compose up --build
```

### Cloud (AWS/GCP/Azure)
```bash
cd deployment/cloud
./deploy.sh
```

See [deployment/README.md](deployment/README.md) for detailed instructions.

---

## 📊 Datasets

### Primary: CodeSearchNet
- **Source:** https://github.com/github/CodeSearchNet
- **Size:** ~2M code samples
- **Languages:** Python, Java, Go, PHP, JavaScript, Ruby
- **License:** Various open-source licenses

### Secondary: Custom Annotations
- GitHub repositories with labeled issues
- Bug reports and fix commits
- Code smell detection datasets

---

## 🧰 Technologies Used

### Machine Learning
- PyTorch / Transformers (Hugging Face)
- CodeBERT, GraphCodeBERT
- TensorBoard for monitoring
- scikit-learn for metrics

### Full-Stack
- **Backend:** FastAPI
- **Frontend:** Streamlit / Gradio
- **Database:** SQLite (for caching)

### Deployment
- Docker / Docker Compose
- Kubernetes (optional)
- AWS / GCP / Azure (cloud deployment)

---

## 📝 Usage Example

```python
from src.deployment.inference import CodeReviewer

# Initialize the model
reviewer = CodeReviewer(model_path="models/best_model.pt")

# Review code
code_snippet = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
"""

results = reviewer.review(code_snippet)

# Output:
# {
#   "issues": [
#     {
#       "type": "bug",
#       "severity": "high",
#       "message": "Potential ZeroDivisionError if list is empty",
#       "line": 5,
#       "confidence": 0.92
#     },
#     {
#       "type": "style",
#       "severity": "low",
#       "message": "Consider using built-in sum() function",
#       "line": 2-4,
#       "confidence": 0.78
#     }
#   ]
# }
```

---

## 🎥 Demo Video

Watch our 10-minute project demo: [presentation/demo-video.mp4](presentation/demo-video.mp4)

**Topics covered:**
- Problem statement and motivation
- Data exploration and preprocessing
- Model architecture and training
- Ablation studies and hyperparameter tuning
- Live application demo
- Deployment pipeline
- Results and future work

---

## 📄 Final Report

See [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md) for the complete academic report including:
- Introduction and related work
- Data description and preprocessing
- Methodology and model architecture
- Experiments and ablation studies
- Results and visualizations
- Conclusion and future work

---

## 🤝 Contributing

This is an academic project. For questions or collaboration:
- Open an issue on GitHub
- Contact team members via university email

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Microsoft Research:** CodeBERT pre-trained model
- **GitHub:** CodeSearchNet dataset
- **Hugging Face:** Transformers library
- **CS 5590 Course Staff:** Guidance and feedback

---

## 📞 Contact

For questions about this project:
- **Repository:** https://github.com/YOUR_USERNAME/code-review-assistant
- **Course:** CS 5590 - AI/ML and Data Science
- **Semester:** Fall 2024

---

**Built with ❤️ for better code quality**

# CRISP-DM Methodology Documentation

## Overview

This document details how the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology was applied to the AI-Powered Code Review Assistant project.

CRISP-DM consists of six phases that provide a structured approach to data mining and machine learning projects:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

---

## Phase 1: Business Understanding

### Objectives
- **Goal:** Automate code review to identify bugs, security vulnerabilities, code smells, style issues, and performance problems
- **Success Criteria:**
  - F1-score ≥ 0.85 for multi-label classification
  - Inference time < 2 seconds per code file
  - Deployable web application with >90% uptime

### Problem Definition
Manual code review is time-consuming, requires expertise, and can miss subtle issues. An automated system can:
- Provide instant feedback to developers
- Catch common mistakes early
- Supplement human reviewers
- Standardize code quality across teams

### Stakeholders
- **Primary Users:** Software developers
- **Secondary Users:** Code reviewers, DevOps teams, educators
- **Project Team:** 4 members with roles in data engineering, ML modeling, full-stack development, and deployment

---

## Phase 2: Data Understanding

### Data Sources
1. **CodeSearchNet Dataset**
   - Source: GitHub/Microsoft Research
   - Size: ~2M code samples
   - Languages: Python, Java, Go, PHP, JavaScript, Ruby
   - Format: JSON with code snippets and documentation

2. **Synthetic Labels** (Initial Version)
   - Created using heuristic rules
   - Categories: bugs, security, code_smell, style, performance

### Exploratory Data Analysis
**Notebook:** [01-EDA.ipynb](../notebooks/01-EDA.ipynb)

Key findings:
- Average code length: 15-30 lines
- Language distribution: Python (40%), JavaScript (35%), others (25%)
- Label imbalance: Style issues most common, security issues least common
- Data quality: High (pre-cleaned by CodeSearchNet team)

---

## Phase 3: Data Preparation

### Preprocessing Steps
**Notebook:** [02-preprocessing.ipynb](../notebooks/02-preprocessing.ipynb)

1. **Tokenization:** CodeBERT tokenizer with max_length=512
2. **Label Encoding:** Multi-hot encoding for 5 issue types
3. **Data Cleaning:**
   - Remove excessive whitespace
   - Normalize indentation
   - Handle edge cases (empty code, very long functions)

4. **Dataset Splitting:**
   - Train: 70% (~420K samples)
   - Validation: 15% (~90K samples)
   - Test: 15% (~90K samples)
   - **Stratification:** Ensured balanced label distribution across splits

### Data Augmentation
To improve model robustness:
- **Variable renaming:** Random variable name changes
- **Format variations:** Different spacing/indentation styles
- **Comment manipulation:** Adding/removing comments
- **Probability:** 30% chance per sample

**Justification:** Prevents overfitting to specific coding styles, improves generalization

---

## Phase 4: Modeling

### Model Selection
**Notebook:** [03-model-training.ipynb](../notebooks/03-model-training.ipynb)

#### Primary Model: CodeBERT
- **Architecture:** 12 transformer layers, 768 hidden dimensions
- **Pre-training:** Trained on 2.1M code-text pairs
- **Fine-tuning:** Multi-label classification head

#### Alternative Models (Ablation Studies)
1. **GraphCodeBERT:** Adds data flow information
2. **SimpleLSTM:** Bidirectional LSTM baseline
3. **DistilCodeBERT:** Distilled version for speed

### Hyperparameter Choices

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Loss Function | BCEWithLogitsLoss | Multi-label classification requires independent probabilities |
| Optimizer | AdamW | Decoupled weight decay prevents overfitting |
| Learning Rate | 2e-5 | Standard for BERT fine-tuning |
| Batch Size | 32 | Balances GPU memory and gradient stability |
| Epochs | 15 | Sufficient with early stopping |
| Dropout | 0.1 | Regularization without losing capacity |
| Activation | GELU | Smoother gradients, standard in transformers |

---

## Phase 5: Evaluation

### Metrics
**Notebook:** [04-evaluation.ipynb](../notebooks/04-evaluation.ipynb)

#### Primary Metrics
- **F1-Score (Macro/Micro):** Overall performance
- **Precision/Recall (per-class):** Class-specific performance
- **ROC-AUC:** Discrimination ability
- **Hamming Loss:** Multi-label specific metric

#### Visualizations (20% of Project)
1. Training/validation loss curves
2. ROC curves per class
3. Precision-Recall curves
4. Confusion matrices
5. TensorBoard dashboards
6. Model comparison charts

### Ablation Studies

| Experiment | F1-Score | Inference Time |
|------------|----------|----------------|
| CodeBERT (baseline) | 0.87 | 1.2s |
| GraphCodeBERT | 0.89 | 1.5s |
| SimpleLSTM | 0.72 | 0.3s |
| DistilCodeBERT | 0.84 | 0.6s |
| No augmentation | 0.82 | 1.2s |

**Key Insights:**
- GraphCodeBERT performs best but is slower
- Data augmentation improves F1 by +0.05
- SimpleLSTM fast but less accurate

---

## Phase 6: Deployment

### Production System
1. **Model Serving:** FastAPI backend
2. **User Interface:** Streamlit web app
3. **Containerization:** Docker
4. **Orchestration:** Docker Compose / Kubernetes

### Deployment Pipeline
1. Train model on GPU (Colab/local)
2. Export best checkpoint
3. Build Docker image
4. Deploy to cloud (AWS/GCP/Azure)
5. Monitor with TensorBoard and logs

### Production Considerations
- **Inference Speed:** <2s requirement met
- **Scalability:** Horizontal scaling with load balancer
- **Monitoring:** Logging, error tracking, performance metrics
- **Security:** Input validation, rate limiting

---

## Lessons Learned

### What Worked Well
✅ CodeBERT pre-training provided excellent starting point  
✅ Data augmentation significantly improved generalization  
✅ Early stopping prevented overfitting  
✅ TensorBoard invaluable for monitoring  

### Challenges
⚠️ Label imbalance required careful evaluation  
⚠️ Synthetic labels less accurate than real annotations  
⚠️ Model size vs. inference speed tradeoff  

### Future Improvements
1. **Real labels:** Collect actual bug reports and fixes
2. **Active learning:** Iteratively improve with user feedback
3. **Multi-language:** Extend to more programming languages
4. **Explainability:** Add attention visualization for interpretability

---

## Iterative Process

CRISP-DM is iterative. We cycled through phases multiple times:

**Iteration 1:** Baseline with synthetic labels  
**Iteration 2:** Added data augmentation (improved F1)  
**Iteration 3:** Hyperparameter tuning (optimized learning rate)  
**Iteration 4:** Model comparison (selected GraphCodeBERT variant)  

Each iteration refined our approach based on evaluation results.

---

## Conclusion

CRISP-DM provided a structured framework that ensured:
- Clear objectives and success criteria
- Thorough data understanding
- Systematic model development
- Rigorous evaluation
- Production-ready deployment

The methodology's flexibility allowed us to iterate and improve while maintaining focus on business objectives.

# Arabic Question Answering System using NLP Techniques

🚀 **A smart Arabic QA system focused on the medical domain**, built with a hybrid ranking model and tiered answer extraction. It directly answers Arabic health-related questions using advanced NLP techniques.

## 🧠 Problem Statement

- Arabic is a low-resource language in NLP.
- Complex morphology and syntax make it hard to process.
- Users seeking medical info need accurate, concise answers.
- Traditional keyword-based search does not satisfy these needs.

## 🎯 Project Goal

Develop an Arabic QA system that retrieves **precise answers** to medical questions in Arabic through:

- Arabic-specific text preprocessing
- Hybrid document ranking (BM25 + NER-TFIDF + Cosine Similarity)
- Tiered answer extraction using deep learning models (e.g., XLM-RoBERTa)
- Clean, intuitive Arabic web interface

## 🏗️ System Architecture

1. **Data Collection**  
   - Source: [DailyMedicalInfo](http://dailymedicalinfo.com)  
   - 600 articles across 30 medical categories  
   - Cleaned and saved in CSV format  

2. **Arabic Text Preprocessing**  
   - Normalization (remove numbers, accents, symbols)  
   - Tokenization & stopword removal  
   - Stemming (root extraction for Arabic words)  

3. **Hybrid Document Ranking**  
   - **BM25** for statistical relevance  
   - **TFIDF + NER + Cosine Similarity** for contextual matching  
   - Multi-tier dynamic ranking pipeline  

4. **Answer Extraction Strategy**  
   - **Top-ranked docs** → XLM-RoBERTa fine-tuned on SQuAD 2.0  
   - **Middle-ranked** → BERT + NER + TFIDF  
   - **Lower-ranked** → Traditional pattern matching  

5. **User Interface**  
   - Arabic web UI built with Flask, HTML, CSS, JavaScript  
   - Input field + 3 answer boxes + source link boxes  

## 📊 Evaluation Metrics

| Metric    | Score |
|-----------|-------|
| Precision | 79%   |
| Recall    | 90%   |
| Accuracy  | 77%   |
| F1 Score  | 84%   |

✅ **Delivers top 3 answers** with verified sources  
✅ **Supports Arabic-speaking users** with non-technical backgrounds

---

## 🔧 Installation & Running

### 📦 Required Libraries

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openpyxl
pip install transformers
pip install nltk
pip install scikit-learn

Run the App: 
python app.py

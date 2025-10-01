# 📜 NLP Contract Summarization & Risk-Sensitive Clause Revision

This project explores **NLP-based approaches** for **contract summarization** and **risk-sensitive clause revision**, leveraging transformer-based models to help lawyers and organizations efficiently review complex contracts.

---

## 🚀 Problem Statement

Contracts contain **thousands of clauses** with varying risk levels. Manual review is time-consuming and error-prone. This project aims to:

* Extract **key obligations, risks, and liabilities** from contracts.
* Generate **summaries** for faster contract understanding.
* Flag and suggest **revisions** for risky clauses.

---

## 📚 Literature Review

We studied several datasets and benchmarks for legal NLP tasks:

* **CUAD (Contract Understanding Atticus Dataset)** → Clause-span extraction with ~13K expert annotations.
* **MAUD (Merger Agreement Understanding Dataset)** → Question-answering on merger agreements.
* **LexGLUE / LEDGAR** → Large-scale provision classification from SEC filings.
* **LegalBench** → Benchmark for legal reasoning tasks.
* **ACORD** → Clause retrieval dataset for drafting workflows.
* **Other works** explored party-specific summarization, abstractive summarization of obligations, and smart contract summarization.

📖 References:

* [CUAD Paper](https://arxiv.org/abs/2103.06268) | [CUAD GitHub](https://github.com/TheAtticusProject/cuad)
* [MAUD](https://arxiv.org/html/2301.00876)
* [LexGLUE](https://aclanthology.org/2022.acl-long.297.pdf)
* [LegalBench](https://arxiv.org/abs/2308.11462)

---

## 📂 Dataset

We primarily used **CUAD**, consisting of:

* **510 commercial contracts**.
* **13K expert-annotated clauses** across **41 clause types** (e.g., *Termination, Liability, Indemnification*).
* Preprocessed into a **2-column dataset**: `[clause, label]`.

---

## ⚙️ Preprocessing

* **PDF extraction** using `pdfplumber`.
* **Segmentation** into clauses/paragraphs.
* **Cleaning**: removal of headers/footers, whitespace normalization.
* **Label mapping**: clause text → contract clause type.

---

## 🧠 Model Architecture

* **Base Model**: [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased).
* **Clause Classification**: maps clause text → clause label.
* **Summarization**: abstractive models (BART/PEGASUS fine-tuned).
* **Risk-Sensitive Revision**: generative LLM prompts to suggest safer clause rewrites.

---

## 📊 Training & Evaluation

* Fine-tuned **Legal-BERT** on the clause-label dataset.
* Metrics: **Accuracy, F1 Score**.
* Results:

  * Contract summarization reduces manual review time.
  * Legal-BERT effectively classifies complex clauses.
  * Risk-sensitive revisions provide actionable insights for lawyers.

---

## 🔮 Interpretation & Impact

* **Efficiency**: Faster contract review process.
* **Risk Management**: Highlights clauses that may expose organizations to liability.
* **Practical Use**: Can be integrated into **contract management systems** or **legal AI assistants**.

# 📊 Evaluation and Fine-Tuning with GPT-2

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-red)](#)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-green)](#)

This repository demonstrates how to **evaluate and fine-tune GPT-2** for text tasks using Hugging Face Transformers and PyTorch.  
It includes a simple evaluator, fine-tuning capability, and unit tests.

---

## 📖 Contents

- `text_evaluator.py` – Core implementation of evaluation and fine-tuning  
- `test_text_evaluator.py` – Unit tests for evaluation and model saving  
- `requirements.txt` – Python dependencies  
- `readme.md` – Documentation and usage guide  
- `url.txt` – Repository link  

---

## 🚀 How to Use

### 1) Clone this repository

```bash
git clone https://github.com/angseesiang/Evaluation-and-fine-tuning.git
cd Evaluation-and-fine-tuning
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate      # On Windows
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the evaluator

```bash
python text_evaluator.py
```

This will evaluate a sample text and print the **evaluation loss**.  
Fine-tuning and saving the model are also demonstrated in the script.

### 5) Run unit tests

```bash
python -m unittest test_text_evaluator.py
```

This verifies that evaluation works and that the model can be saved successfully.

---

## 🛠️ Requirements

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Datasets
- PyYAML
- tf-keras

All dependencies are listed in `requirements.txt`.

---

## 📌 Notes

- The evaluator computes a **loss value** for given input text using GPT-2.  
- You can fine-tune the model on your own dataset with `fine_tune()`.  
- Models and tokenizers can be saved using `save_model(path)` for reuse.  

---

## 📜 License

This project is for **educational purposes only**.

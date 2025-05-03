# NLP Lung Cancer RECIST Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, end-to-end Natural Language Processing (NLP) pipeline for extracting and evaluating treatment response (RECIST) endpoints from lung cancer clinical notes. This repository provides data processing, model training, inference, evaluation, and a lightweight web API for demonstration.

---

## 🚀 Features

- **Data Ingestion & Preprocessing**  
  Load raw clinical notes, apply cleaning, tokenization, and formatting for downstream NLP tasks.

- **Model Training & Inference**  
  Train custom NER and classification models; serialize trained weights under `models/`.

- **Evaluation Suite**  
  Standard metrics (precision, recall, F1) and custom RECIST-style reporting under `eval/`.

- **Reusable Utilities**  
  Helper functions for data loading, metric computation, and result visualization in `utils/`.

- **Scripts & Automation**  
  Command-line entry points to run each stage of the pipeline in `scripts/`.

- **Demo API**  
  A minimal Flask (or FastAPI) app (`app.py`) that exposes the model as a RESTful service.

---

## 📂 Repository Structure

\`\`\`
.
├── eval/                  # Evaluation scripts & reports
│   ├── evaluate.py
│   └── reports/
├── models/                # Trained model checkpoints
│   ├── ner_model.pt
│   └── classifier_model.pt
├── scripts/               # Standalone scripts for each pipeline stage
│   ├── preprocess.py
│   ├── train_ner.py
│   ├── train_classifier.py
│   └── infer.py
├── utils/                 # Utility modules
│   ├── data_loader.py
│   ├── metrics.py
│   └── text_processing.py
├── .gitignore
├── README.md
├── requirements.txt       # Python dependencies
├── app.py                 # Demo REST API for inference
└── run_pipeline.py        # Orchestrator: runs full pipeline end-to-end
\`\`\`

---

## ⚙️ Installation

1. **Clone the repo**  
   \`\`\`bash
   git clone https://github.com/PittNAIL/nlp-lc-recist.git
   cd nlp-lc-recist
   \`\`\`

2. **Create & activate a virtual environment**  
   \`\`\`bash
   python3 -m venv venv
   source venv/bin/activate
   \`\`\`

3. **Install dependencies**  
   \`\`\`bash
   pip install --upgrade pip
   pip install -r requirements.txt
   \`\`\`

---

## 💡 Usage

### 1. Run Full Pipeline

Use the orchestrator to go from raw data → predictions → evaluation in one command:

\`\`\`bash
python run_pipeline.py \
  --input_dir data/raw/ \
  --output_dir outputs/ \
  --config config/pipeline.yaml
\`\`\`

### 2. Individual Stages

- **Preprocess**  
  \`\`\`bash
  python scripts/preprocess.py \
    --input data/raw/notes.jsonl \
    --output data/processed/notes_tok.jsonl
  \`\`\`

- **Train NER**  
  \`\`\`bash
  python scripts/train_ner.py \
    --train data/processed/train.jsonl \
    --dev data/processed/dev.jsonl \
    --output models/ner_model.pt
  \`\`\`

- **Train Classifier**  
  \`\`\`bash
  python scripts/train_classifier.py \
    --features data/processed/features.npz \
    --labels data/processed/labels.npy \
    --output models/classifier_model.pt
  \`\`\`

- **Inference**  
  \`\`\`bash
  python scripts/infer.py \
    --model models/ner_model.pt \
    --input data/processed/test.jsonl \
    --output predictions/ner_preds.jsonl
  \`\`\`

- **Evaluate**  
  \`\`\`bash
  python eval/evaluate.py \
    --predictions predictions/ \
    --gold data/processed/test_gold.jsonl \
    --report eval/reports/metrics.json
  \`\`\`

### 3. Launch Demo API

Start the REST API for on-the-fly inference:

\`\`\`bash
python app.py --host 0.0.0.0 --port 5000
\`\`\`

- **Endpoint**  
  \`POST /predict\`  
  **Payload**  
  \`\`\`json
  { "text": "...clinical note text..." }
  \`\`\`  
  **Response**  
  \`\`\`json
  {
    "entities": [ { "start": 10, "end": 25, "label": "TUMOR_SIZE", ... } ],
    "recist_call": "Stable Disease"
  }
  \`\`\`

---

## 📝 Configuration

All hyperparameters and filepaths can be set via the top-level \`config/pipeline.yaml\`. Example:

\`\`\`yaml
preprocessing:
  lowercase: true
  remove_pii: true

ner:
  learning_rate: 3e-5
  batch_size: 16
  epochs: 10

classifier:
  hidden_dim: 256
  dropout: 0.1

paths:
  raw_data: data/raw/
  processed_data: data/processed/
  model_dir: models/
  output_dir: outputs/
\`\`\`

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (\`git checkout -b feature/my-new-feature\`)  
3. Commit your changes (\`git commit -am 'Add feature'\`)  
4. Push to the branch (\`git push origin feature/my-new-feature\`)  
5. Open a Pull Request  

Please follow the existing code style and add tests where appropriate.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

Sonish Sivarajkumar  
– PhD Candidate, PittNAIL Lab, University of Pittsburgh  
– ✉️ sonish.sivarajkumar@pitt.edu  

Feel free to open issues or reach out with questions!

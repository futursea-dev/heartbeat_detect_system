# Heartbeat Analysis with LLMs and Summarization

This project is a Python-based system designed to analyze heartbeat audio recordings and provide diagnostic assistance using state-of-the-art AI models. It extracts audio features from heartbeat `.wav` files, generates diagnostic insights using the `microsoft/phi-2` large language model, and summarizes those insights with the `sshleifer/distilbart-cnn-12-6` summarization model — all running locally without any need for API keys.

## Features

- Extract heartbeat features such as MFCC coefficients, tempo, and duration from audio recordings.
- Generate diagnostic observations using the `microsoft/phi-2` LLM based on extracted audio features.
- Summarize the generated diagnostics with the `sshleifer/distilbart-cnn-12-6` summarization model.
- Can be integrated into web frameworks like Django for interactive user interfaces.
- Fully offline and self-contained, relying on Hugging Face transformers and PyTorch.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/heartbeat-analysis.git
   cd heartbeat-analysis

2. Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

4. Download necessary Hugging Face models (run this once):
    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    AutoTokenizer.from_pretrained("microsoft/phi-2")
    AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

    pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

## Usage

1. Prepare your heartbeat .wav audio files.

2. Run feature extraction, diagnosis generation, and summarization:
    ```python
    from audio_analysis.heartbeat_analyzer import extract_heartbeat_features
    from llm.diagnostic_assistant import get_diagnosis_from_features
    from llm.summarizer import summarize_diagnosis

    wav_path = "path/to/heartbeat.wav"
    features = extract_heartbeat_features(wav_path)
    diagnosis = get_diagnosis_from_features(features)
    summary = summarize_diagnosis(diagnosis)

    print("Diagnostic Report:", diagnosis)
    print("Summary:", summary)

## Django Integration
The system can be easily integrated with Django to build a web interface for uploading heartbeat audio and displaying analysis results.

## Contributing
Contributions and suggestions are welcome! Feel free to open issues or pull requests.
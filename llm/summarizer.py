from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    tokenizer="sshleifer/distilbart-cnn-12-6",
    # cache_dir="./distilbart-cnn-12-6"
)

def summarize_medical_record(text):
    if not text or len(text.strip()) < 10:
        return "Not enough content to summarize."

    try:
        summary = summarizer(
            text,
            max_length=min(150, len(text.split()) + 20),
            min_length=10,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization failed: {str(e)}"

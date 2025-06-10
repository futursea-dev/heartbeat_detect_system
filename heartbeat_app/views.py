from django.shortcuts import render
from .forms import UploadForm
from audio_analysis.heartbeat_analyzer import extract_heartbeat_features, load_audio_waveform
from llm.summarizer import summarize_medical_record
from llm.diagnostic_assistant import get_diagnosis_from_features
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for rendering plots
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Ensure the figure is tight
    plt.close()  # Prevent showing or reusing the plot
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def index(request):
    result = None
    waveform_img = None
    mfcc_img = None

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                text = form.cleaned_data['medical_text']
                file = request.FILES['audio_file']

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    for chunk in file.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name

                print("Saved audio to:", tmp_path)

                features = extract_heartbeat_features(tmp_path)
                print("Extracted features:", features)

                diagnosis = get_diagnosis_from_features(features)
                print("Diagnosis:", diagnosis)

                # diagnosis_map = {
                #     "normal": "Normal heartbeat — no issues detected.",
                #     "murmur_detected": "Possible heart murmur detected. Consider further evaluation.",
                #     "irregular": "Irregular heartbeat — abnormal rhythm detected.",
                #     "unknown": "Could not determine — please upload clearer audio."
                # }

                # diagnosis = diagnosis_map.get(diagnosis, "Unknown result.")

                summary = summarize_medical_record(text)
                print("Summary:", summary)

                # Plot Waveform
                waveform = load_audio_waveform(tmp_path)
                plt.figure(figsize=(10, 3))
                plt.plot(waveform)
                plt.title('Heartbeat Audio Waveform')
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')
                waveform_img = plot_to_base64()

                print(len(waveform_img), "samples in waveform")

                # Plot MFCC mean values as a bar plot
                mfcc_mean = features.get('mfcc_mean', [])
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(mfcc_mean)), mfcc_mean)
                plt.title('MFCC Mean Coefficients')
                plt.xlabel('Coefficient Index')
                plt.ylabel('Value')
                mfcc_img = plot_to_base64()
                
                # Format MFCC mean
                mfcc_readable = ", ".join([f"{x:.2f}" for x in features["mfcc_mean"]])

                # Tempo: take first value if array
                tempo = features["tempo"][0] if isinstance(features["tempo"], (list, np.ndarray)) else features["tempo"]

                readable_features = {
                    "MFCC Mean": mfcc_readable,
                    "Tempo (BPM)": f"{tempo:.1f}",
                    "Duration (sec)": f"{features['duration']:.2f}"
                }

                result = {
                    'features': readable_features,
                    'diagnosis': diagnosis,
                    'summary': summary
                }

            except Exception as e:
                print("🔥 ERROR:", e)
                result = {'error': str(e)}
    else:
        form = UploadForm()
    return render(request, 'heartbeat_app/index.html', {'form': form, 'result': result, 'waveform_img': waveform_img, 'mfcc_img': mfcc_img})

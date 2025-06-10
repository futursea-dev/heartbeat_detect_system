from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

def get_diagnosis_from_features(features):
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

    mfcc = np.array(features['mfcc_mean'])
    tempo = features['tempo'] if isinstance(features['tempo'], (list, np.ndarray)) else features['tempo']
    duration = features['duration']

    if tempo < 50 or tempo > 130:
        return "Abnormal heartbeat: Unusually low or high tempo detected."
    
    if mfcc[0] < -500:
        return "Possible murmur: Low-frequency energy is strong."
    
    if duration < 2.0:
        return "Audio too short for reliable diagnosis."
    
    return "Normal heartbeat."

#     prompt = f"""You are a healthcare assistant. Based on the following heartbeat features:
# { features }
# Suggest any abnormalites or possible cardiac conditions, and provide a brief medical summary.
# """
    
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_new_tokens=150)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
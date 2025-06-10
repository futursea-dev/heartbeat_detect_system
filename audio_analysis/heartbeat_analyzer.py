import librosa
# import numpy as np
def load_audio_waveform(path, sr=22050):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def extract_heartbeat_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {
        "mfcc_mean": mfcc.mean(axis=1).tolist(),
        "tempo": tempo, 
        "duration": librosa.get_duration(y=y, sr=sr),
    }
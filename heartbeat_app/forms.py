from django import forms

class UploadForm(forms.Form):
    audio_file = forms.FileField(label='Upload Heartbeat Audio WAV File')
    medical_text = forms.CharField(widget=forms.Textarea, label='Medical Notes')

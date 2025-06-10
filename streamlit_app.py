import streamlit as st
import torch
import librosa
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# โหลดโมเดลจาก Hugging Face
MODEL_NAME = "B4by/Test"
model = AutoModelForSpeechSeq2Seq.from_pretrained("B4by/Test", use_auth_token=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# ฟังก์ชันแปลงเสียง
def convert_audio(input_audio_path):
    # โหลดเสียง
    waveform, sample_rate = librosa.load(input_audio_path, sr=16000)
    
    # ประมวลผลเสียงด้วยโมเดล
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    # แปลงผลลัพธ์กลับเป็นเสียง
    output_waveform = processor.batch_decode(outputs, return_tensors="pt").squeeze().numpy()
    output_audio_path = "converted_audio.wav"
    sf.write(output_audio_path, output_waveform, sample_rate)

    return output_audio_path

# UI ด้วย Streamlit
st.title("AI Voice Anime Converter")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    # บันทึกไฟล์ชั่วคราว
    input_audio_path = f"temp_{uploaded_file.name}"
    with open(input_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ประมวลผลเสียง
    output_audio_path = convert_audio(input_audio_path)
    
    # แสดงผลลัพธ์
    st.audio(output_audio_path, format="audio/wav")
    st.download_button("Download Converted Audio", output_audio_path)


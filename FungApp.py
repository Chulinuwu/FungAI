import streamlit as st
import pandas as pd
import numpy as np
import librosa
import sounddevice as sd
import time
import tempfile
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.io.wavfile import write
import io
from io import BytesIO
from matplotlib.figure import Figure
import datetime
import base64

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Fung Music Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ฟังก์ชั่นสำหรับสกัด features จากไฟล์เสียง
def extract_features_from_audio(audio, sr=22050):
    """สกัดคุณลักษณะเสียงจากข้อมูลเสียง"""
    features = {}
    
    # Basic properties
    features['duration'] = len(audio) / sr
    
    # Spectral features (mean and std)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
    
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
    features['spectral_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
    features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    
    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features['tempo'] = float(tempo)
    
    # Energy
    features['rms_mean'] = np.mean(librosa.feature.rms(y=audio)[0])
    
    return features

# ฟังก์ชั่นสำหรับค้นหาเพลงที่คล้ายกัน
def find_similar_song(recorded_features, db_features, n_results=5):
    """หาเพลงที่คล้ายที่สุดจากฐานข้อมูล"""
    
    # แปลงคุณลักษณะเสียงให้เป็น DataFrame
    recorded_df = pd.DataFrame([recorded_features])
    
    # เลือกเฉพาะคอลัมน์ที่ต้องการเปรียบเทียบ (ไม่รวมชื่อไฟล์)
    feature_cols = [col for col in db_features.columns if col != 'filename']
    
    # Normalize ข้อมูล
    scaler = StandardScaler()
    db_features_scaled = scaler.fit_transform(db_features[feature_cols])
    recorded_features_scaled = scaler.transform(recorded_df[feature_cols])
    
    # คำนวณความคล้ายคลึง (ระยะห่าง) ระหว่างเสียงที่บันทึกกับทุกเพลงในฐานข้อมูล
    distances = []
    for i, row in enumerate(db_features_scaled):
        # ใช้ cosine distance (1 - cosine similarity)
        dist = cosine(recorded_features_scaled[0], row)
        distances.append((i, dist))
    
    # เรียงลำดับตามความคล้ายคลึง (น้อยไปมาก = คล้ายมากไปคล้ายน้อย)
    distances.sort(key=lambda x: x[1])
    
    # คืนค่าเพลงที่คล้ายที่สุด n_results เพลง
    similar_songs = []
    for i in range(min(n_results, len(distances))):
        idx, dist = distances[i]
        song_name = db_features.iloc[idx]['filename']
        similarity_score = 1 - dist  # แปลงเป็นคะแนนความคล้าย (0-1)
        similar_songs.append((song_name, similarity_score))
    
    return similar_songs

# ฟังก์ชั่นสำหรับเทรนโมเดล
@st.cache_resource
def train_song_recognition_model(df_features):
    """สร้างโมเดล ML สำหรับจำแนกเพลง"""
    
    # แยก features และ target (ชื่อเพลง)
    X = df_features.drop('filename', axis=1)
    y = df_features['filename']
    
    # Encode ชื่อเพลงเป็นตัวเลข
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # แบ่งข้อมูลสำหรับ train และ test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # สร้าง Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # ทดสอบโมเดล
    accuracy = model.score(X_test, y_test)
    
    return model, le, accuracy

# ฟังก์ชั่นสำหรับทำนายเพลงด้วยโมเดล ML
def predict_song_with_model(audio, model, label_encoder, df_features):
    """ทำนายเพลงโดยใช้โมเดล ML"""
    
    # สกัดคุณลักษณะเสียง
    features = extract_features_from_audio(audio)
    
    # แปลงเป็น DataFrame และเรียงคอลัมน์ให้ตรงกับตอนเทรน
    features_df = pd.DataFrame([features])
    X = features_df[df_features.drop('filename', axis=1).columns]
    
    # ทำนายเพลง
    y_pred = model.predict_proba(X)
    
    # รับค่าความน่าจะเป็นและชื่อเพลงที่เป็นไปได้มากที่สุด 5 อันดับ
    top_indices = np.argsort(y_pred[0])[::-1][:5]
    top_probs = y_pred[0][top_indices]
    top_songs = label_encoder.inverse_transform(top_indices)
    
    results = [(song, prob) for song, prob in zip(top_songs, top_probs)]
    return results

# ฟังก์ชั่นสำหรับบันทึกเสียง
def record_audio(duration=10, fs=22050):
    """บันทึกเสียงจากไมโครโฟน"""
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    
    # สร้าง progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # แสดงความคืบหน้าในการบันทึก
    for i in range(100):
        # ปรับปรุง progress bar
        progress_bar.progress(i + 1)
        status_text.text(f"กำลังบันทึกเสียง... {i+1}%")
        time.sleep(duration/100)
    
    sd.wait()  # รอจนกว่าการบันทึกจะเสร็จ
    status_text.text("บันทึกเสียงเสร็จสิ้น!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return audio.flatten()

# ฟังก์ชั่นสำหรับบันทึกไฟล์เสียงลงในไฟล์
def save_audio_to_file(audio, sr=22050):
    """บันทึกเสียงลงในไฟล์ .wav"""
    # สร้างไฟล์ชั่วคราว
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_filename = temp_file.name
    
    # บันทึกไฟล์เสียง
    scaled_audio = np.int16(audio * 32767)  # แปลงเป็น 16-bit PCM
    write(temp_filename, sr, scaled_audio)
    
    return temp_filename

# ฟังก์ชั่นสำหรับสร้างกราฟแสดงเปรียบเทียบผลลัพธ์
def create_results_chart(results):
    """สร้างกราฟแสดงผลลัพธ์การทำนาย"""
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    
    songs = [result[0].split('.')[0][:25] + '...' if len(result[0].split('.')[0]) > 25 else result[0].split('.')[0] for result in results]
    probs = [result[1] for result in results]
    
    bars = ax.barh(songs, probs, color='skyblue')
    ax.set_xlabel('Probability')
    ax.set_title('Top 5 Most Similar Songs')
    ax.set_xlim(0, 1.0)
    
    # เพิ่มเปอร์เซ็นต์ที่แท่งกราฟ
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f'{prob*100:.1f}%', 
                ha='left', va='center', color='black', fontweight='bold')
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    
    return buf

# ฟังก์ชั่นสำหรับแสดง waveform ของเสียงที่บันทึก
def plot_waveform(audio, sr=22050):
    """สร้างกราฟแสดง waveform ของเสียงที่บันทึก"""
    fig = Figure(figsize=(10, 3))
    ax = fig.subplots()
    
    # Plot waveform
    times = np.arange(len(audio)) / sr
    ax.plot(times, audio, color='blue', alpha=0.7)
    
    # ตกแต่งกราฟ
    ax.set_title('Waveform ของเสียงที่บันทึก')
    ax.set_xlabel('เวลา (วินาที)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim([-1, 1])
    ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    
    return buf

# ฟังก์ชั่นสำหรับแสดง spectrogram ของเสียงที่บันทึก
def plot_spectrogram(audio, sr=22050):
    """สร้าง spectrogram ของเสียงที่บันทึก"""
    fig = Figure(figsize=(10, 3))
    ax = fig.subplots()
    
    # คำนวณ spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    
    # Plot spectrogram
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Spectrogram ของเสียงที่บันทึก')
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.f dB")
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    
    return buf

# Main App
def main():
    st.title("🎵 Fung Music Recognition System 🎵")
    st.markdown("### ระบบรู้จำเพลงอัตโนมัติ (คล้าย Shazam) จากฐานข้อมูลเพลง Serious Bacon")
    
    # โหลดฐานข้อมูล features
    try:
        df_features = pd.read_csv('audio_features.csv')
        st.success(f"โหลดฐานข้อมูลสำเร็จ! มีเพลงทั้งหมด {len(df_features)} เพลง")
    except Exception as e:
        st.error(f"ไม่สามารถโหลดฐานข้อมูลได้: {e}")
        st.stop()
    
    # สร้าง sidebar
    st.sidebar.header("🔧 ตั้งค่าระบบ")
    
    recognition_method = st.sidebar.selectbox(
        "เลือกวิธีการจำแนกเพลง",
        ["Similarity Search", "Machine Learning Model"]
    )
    
    duration = st.sidebar.slider("ระยะเวลาในการบันทึกเสียง (วินาที)", 3, 20, 10)
    
    if recognition_method == "Machine Learning Model":
        with st.sidebar.expander("🧠 โมเดล Machine Learning", expanded=True):
            train_button = st.button("เทรนโมเดล", key="train_model")
            
            if train_button or "model" not in st.session_state:
                with st.spinner("กำลังเทรนโมเดล..."):
                    model, le, accuracy = train_song_recognition_model(df_features)
                    st.session_state.model = model
                    st.session_state.le = le
                    st.session_state.model_accuracy = accuracy
                st.sidebar.success(f"เทรนโมเดลเสร็จสิ้น! ความแม่นยำ: {accuracy:.2f}")
    
    # แสดงรายชื่อเพลงในฐานข้อมูล
    with st.sidebar.expander("📋 รายชื่อเพลงในฐานข้อมูล", expanded=False):
        for i, song in enumerate(df_features['filename'], 1):
            st.write(f"{i}. {song}")
    
    # สร้างแท็บ
    tab1, tab2 = st.tabs(["🎤 รู้จำเพลง", "📊 ข้อมูลเพิ่มเติม"])
    
    with tab1:
        # ส่วนการรู้จำเพลง
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔊 บันทึกเสียง")
            record_button = st.button("บันทึกเสียง", key="record_audio")
            
            # ถ้ากดปุ่มบันทึกเสียง
            if record_button:
                # แสดงนับถอยหลัง
                countdown_placeholder = st.empty()
                for i in range(3, 0, -1):
                    countdown_placeholder.markdown(f"<h1 style='text-align: center;'>เริ่มบันทึกใน... {i}</h1>", unsafe_allow_html=True)
                    time.sleep(1)
                countdown_placeholder.empty()
                
                # บันทึกเสียง
                with st.spinner("กำลังบันทึกเสียง..."):
                    audio = record_audio(duration=duration)
                    st.session_state.audio = audio
                    
                    # บันทึกเสียงลงไฟล์
                    audio_file = save_audio_to_file(audio)
                    st.session_state.audio_file = audio_file
                
                # แสดง waveform
                st.markdown("### 📊 Waveform ของเสียงที่บันทึก")
                waveform_buf = plot_waveform(audio)
                st.image(waveform_buf)
                
                # แสดงตัวเล่นเสียง
                st.markdown("### 🎧 เล่นเสียงที่บันทึก")
                st.audio(audio_file)
                
                # วิเคราะห์และหาเพลงที่คล้ายกัน
                with st.spinner("กำลังวิเคราะห์เสียง..."):
                    if recognition_method == "Similarity Search":
                        features = extract_features_from_audio(audio)
                        similar_songs = find_similar_song(features, df_features)
                        st.session_state.results = similar_songs
                    else:  # Machine Learning Model
                        if "model" in st.session_state:
                            results = predict_song_with_model(audio, st.session_state.model, st.session_state.le, df_features)
                            st.session_state.results = results
                        else:
                            st.error("กรุณาเทรนโมเดลก่อน!")
                            st.session_state.results = []
        
        with col2:
            st.markdown("### 🔍 ผลการค้นหา")
            
            # แสดงผลการค้นหา
            if "results" in st.session_state and st.session_state.results:
                results_buf = create_results_chart(st.session_state.results)
                st.image(results_buf)
                
                st.markdown("### 🏆 อันดับเพลงที่คล้ายที่สุด")
                for i, (song, score) in enumerate(st.session_state.results, 1):
                    st.markdown(f"**{i}. {song}**")
                    st.progress(float(score))
                    st.text(f"ความคล้ายคลึง: {score:.4f}")
            else:
                st.info("กรุณาบันทึกเสียงเพื่อค้นหาเพลง")
    
    with tab2:
        # ข้อมูลเพิ่มเติม
        st.markdown("### 📈 รายละเอียดของฐานข้อมูล")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 สถิติของฐานข้อมูล")
            st.dataframe(df_features.describe())
        
        with col2:
            st.markdown("#### 🎵 กราฟแสดงความเร็วจังหวะ (Tempo) ของเพลง")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            tempo_data = df_features.sort_values('tempo', ascending=False)
            shortened_names = [name.split('.')[0][:20] + '...' if len(name.split('.')[0]) > 20 else name.split('.')[0] for name in tempo_data['filename']]
            
            bars = ax.barh(shortened_names, tempo_data['tempo'], color='skyblue')
            ax.set_xlabel('Tempo (BPM)')
            ax.set_title('Tempo ของแต่ละเพลง')
            
            # เพิ่มค่า tempo ที่แท่งกราฟ
            for bar, tempo in zip(bars, tempo_data['tempo']):
                ax.text(tempo + 1, bar.get_y() + bar.get_height()/2, f'{tempo:.1f}', 
                        ha='left', va='center', fontsize=8)
                
            st.pyplot(fig)

# รัน App
if __name__ == '__main__':
    main()
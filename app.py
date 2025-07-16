import streamlit as st
import pandas as pd
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Instagram Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Global Styles */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #4b6cb7 0%, #182848  100%);
    min-height: 100vh;
    color: white !important;
}

.main {
    background: transparent;
    font-family: 'Poppins', sans-serif;
    color: white !important;
}

/* Override Streamlit's default background */
.stApp {
    background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
    color: white !important;
}

/* Make ALL text white by default */
* {
    color: white !important;
}

/* Custom Header */
.main-header {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    padding: 2rem 1rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.main-header h1 {
    color: white !important;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.2rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Card Containers */
.card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(15px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.25);
    background: rgba(255, 255, 255, 0.15);
}

.card h2 {
    color: white !important;
    font-weight: 600;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
}

.card h3 {
    color: white !important;
    font-weight: 500;
    margin-bottom: 1rem;
}

.card p {
    color: white !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #3f5efb 0%, #2b5876 100%);
    color: white !important;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(108, 92, 231, 0.3);
    margin-bottom: 1rem;
}

.metric-card h3 {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    color: white !important;
}

.metric-card p {
    font-size: 1rem !important;
    margin: 0.5rem 0 0 0 !important;
    opacity: 0.9;
    color: white !important;
}

/* Override Streamlit's metric styling */
.metric-card .stMetric {
    color: white !important;
}

.metric-card .stMetric > div {
    color: white !important;
}

.metric-card .stMetric label {
    color: white !important;
}

/* Upload Area */
.upload-area {
    border: 2px dashed rgba(255,255,255,0.6);
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    margin-bottom: 1.2rem;
}

.upload-area:hover {
    border-color: rgba(255,255,255,1);
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
}

.upload-area h3 {
    color: white !important;
    margin-bottom: 1rem;
    font-weight: 600;
}

.upload-area p {
    color: rgba(255,255,255,0.9) !important;
    margin: 0;
}

.upload-icon {
    font-size: 4rem;
    color: white !important;
    margin-bottom: 1rem;
}

/* UNIVERSAL BUTTON STYLING - SEMUA TOMBOL BIRU DENGAN TULISAN PUTIH */
button,
.stButton > button,
.stDownloadButton > button,
.stDownloadButton button,
div[data-testid="stDownloadButton"] > button,
div[data-testid="stDownloadButton"] button,
[data-testid="stDownloadButton"] button,
.stFileUploader button,
input[type="button"],
input[type="submit"],
button[kind="secondary"],
button[data-testid*="download"],
a[download] {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3) !important;
    font-family: 'Poppins', sans-serif !important;
    text-decoration: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 44px !important;
}

/* Hover state untuk semua tombol */
button:hover,
.stButton > button:hover,
.stDownloadButton > button:hover,
.stDownloadButton button:hover,
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stDownloadButton"] button:hover,
[data-testid="stDownloadButton"] button:hover,
.stFileUploader button:hover,
input[type="button"]:hover,
input[type="submit"]:hover,
button[kind="secondary"]:hover,
button[data-testid*="download"]:hover,
a[download]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: white !important;
}

/* Active state untuk semua tombol */
button:active,
.stButton > button:active,
.stDownloadButton > button:active,
.stDownloadButton button:active,
div[data-testid="stDownloadButton"] > button:active,
div[data-testid="stDownloadButton"] button:active,
[data-testid="stDownloadButton"] button:active,
.stFileUploader button:active,
input[type="button"]:active,
input[type="submit"]:active,
button[kind="secondary"]:active,
button[data-testid*="download"]:active,
a[download]:active {
    transform: translateY(0px) !important;
    box-shadow: 0 3px 10px rgba(59, 130, 246, 0.3) !important;
}

/* Memastikan semua teks dalam tombol berwarna putih */
button *,
.stButton > button *,
.stDownloadButton > button *,
.stDownloadButton button *,
div[data-testid="stDownloadButton"] > button *,
div[data-testid="stDownloadButton"] button *,
[data-testid="stDownloadButton"] button *,
.stFileUploader button *,
input[type="button"] *,
input[type="submit"] *,
button[kind="secondary"] *,
button[data-testid*="download"] *,
a[download] * {
    color: white !important;
}

/* Focus state */
button:focus,
.stButton > button:focus,
.stDownloadButton > button:focus,
.stDownloadButton button:focus,
div[data-testid="stDownloadButton"] > button:focus,
div[data-testid="stDownloadButton"] button:focus,
[data-testid="stDownloadButton"] button:focus,
.stFileUploader button:focus,
input[type="button"]:focus,
input[type="submit"]:focus,
button[kind="secondary"]:focus,
button[data-testid*="download"]:focus,
a[download]:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25) !important;
}

/* Khusus untuk file uploader - tetap biru tapi rounded lebih kecil */
.stFileUploader button {
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.9rem !important;
}

.stFileUploader button:hover {
    transform: translateY(-1px) !important;
}

/* Text Area */
.stTextArea > div > div > textarea {
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 15px;
    padding: 1rem;
    font-size: 1rem;
    font-family: 'Poppins', sans-serif;
    transition: border-color 0.3s ease;
    background: rgba(255, 255, 255, 0.1);
    color: white !important;
    resize: vertical;
}

.stTextArea > div > div > textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.stTextArea > div > div > textarea::placeholder {
    color: rgba(255, 255, 255, 0.6) !important;
}

/* Selectbox */
.stSelectbox > div > div > select {
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 15px;
    padding: 0.8rem;
    font-size: 1rem;
    font-family: 'Poppins', sans-serif;
    background: rgba(255, 255, 255, 0.1);
    color: white !important;
    transition: border-color 0.3s ease;
}

.stSelectbox > div > div > select:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.stSelectbox > div > div > select option {
    background: #2a3e6c !important;
    color: white !important;
}

/* File uploader - Fixed styling for black text */
.stFileUploader {
    background: rgba(255, 255, 255, 0.9);
    border: 2px dashed rgba(0, 0, 0, 0.3);
    border-radius: 15px;
    padding: 1rem;
    backdrop-filter: blur(10px);
}

.stFileUploader:hover {
    border-color: rgba(0, 0, 0, 0.5);
    background: rgba(255, 255, 255, 0.95);
}

.stFileUploader label {
    color: black !important;
    font-weight: 500;
    font-family: 'Poppins', sans-serif;
}

.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.9);
    border: 2px dashed rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    color: black !important;
}

.stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0, 0, 0, 0.5);
    background: rgba(255, 255, 255, 0.95);
}

.stFileUploader [data-testid="stFileUploaderDropzone"] * {
    color: black !important;
}

.stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
    color: black !important;
}

.stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] p {
    color: black !important;
}

.stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] span {
    color: black !important;
}

.stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] small {
    color: black !important;
}

/* Results */
.result-positive {
    background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
    color: white !important;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
}

.result-negative {
    background: linear-gradient(135deg, #d63031 0%, #ff6b6b 100%);
    color: white !important;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(214, 48, 49, 0.3);
}

.result-neutral {
    background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
    color: white !important;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(253, 203, 110, 0.3);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%) !important;
    color: white !important;
    padding: 1.5rem;
    font-family: 'Poppins', sans-serif;
    box-shadow: 4px 0 12px rgba(0,0,0,0.3);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar elements */
.css-1d391kg .stRadio label, .css-1lcbmhc .stRadio label {
    color: white !important;
}

.css-1d391kg .stSelectbox label, .css-1lcbmhc .stSelectbox label {
    color: white !important;
}

.css-1d391kg .stCheckbox label, .css-1lcbmhc .stCheckbox label {
    color: white !important;
}

.css-1d391kg .stMarkdown, .css-1lcbmhc .stMarkdown {
    color: white !important;
}

/* Streamlit elements white text */
.stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: white !important;
}

.stText, .stCaption {
    color: white !important;
}

.stInfo, .stSuccess, .stWarning, .stError {
    color: white !important;
}

.stExpander summary {
    color: white !important;
}

.stExpander [data-testid="stExpanderDetails"] {
    color: white !important;
}

.stDataFrame {
    color: white !important;
}

/* Dataframe styling */
.stDataFrame > div {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px;
}

.stDataFrame table {
    background: rgba(255, 255, 255, 0.05) !important;
    color: white !important;
}

.stDataFrame th {
    background: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border-color: rgba(255,255,255,0.2) !important;
}

.stDataFrame td {
    color: white !important;
    border-color: rgba(255,255,255,0.1) !important;
}

/* Radio buttons */
.stRadio > div {
    color: white !important;
}

.stRadio label {
    color: white !important;
}

/* Checkboxes */
.stCheckbox > div {
    color: white !important;
}

.stCheckbox label {
    color: white !important;
}

/* Metrics */
.stMetric > div {
    color: white !important;
}

.stMetric label {
    color: white !important;
}

/* Spinner */
.stSpinner > div {
    color: white !important;
}

/* Labels */
label {
    color: white !important;
}

/* Override any white backgrounds in main content */
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Fix agar segitiga dropdown (arrow) muncul di selectbox sidebar */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: rgba(255, 255, 255, 0.1);
    color: white !important;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.4);
    font-family: 'Poppins', sans-serif;
    position: relative;
}

[data-testid="stSidebar"] .stSelectbox > div > div::after {
    content: "▼";
    position: absolute;
    right: 15px;
    top: 50%;
    transform: translateY(-50%);
    color: white !important;
    pointer-events: none;
    font-size: 0.8rem;
}

/* Responsive */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .card {
        padding: 1rem;
    }
    
    .metric-card h3 {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)
# === HEADER ===
st.markdown("""
<div class="main-header">
    <h1>🎯 Instagram Sentiment Analysis</h1>
    <p>Analisis sentimen untuk ulasan Instagram menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# === PREPROCESSING (TETAP SAMA) ===
stop_words = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# === LOAD MODEL (TETAP SAMA) ===
@st.cache_resource
def load_model():
    try:
        with open("svm_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'svm_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        st.stop()

# === WORDCLOUD FUNCTION (TETAP SAMA) ===
def generate_wordcloud(text, color="Dark2"):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=color,
        collocations=False,
        font_path=None
    ).generate(text)
    return wc

# === SIDEBAR ===
with st.sidebar:
    st.markdown("## 📋 Navigation")
    analysis_type = st.radio(
        "Pilih jenis analisis:",
        ["📄 Analisis File CSV", "✍️ Analisis Manual"],
        key="analysis_type"
    )
    
    st.markdown("---")
    st.markdown("## 🔧 Settings")
    
    if analysis_type == "📄 Analisis File CSV":
        chart_type = st.selectbox("Pilih jenis chart:", ["Bar Chart", "Pie Chart", "Both"])
        show_wordcloud = st.checkbox("Tampilkan WordCloud", value=True)
    
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Sentiment Analysis Dashboard**
    
    Aplikasi ini menggunakan:
    - SVM (Support Vector Machine)
    - TF-IDF Vectorization
    - Sastrawi Stemmer
    - NLTK Stopwords
    """)

# Load model
model, vectorizer = load_model()

# === MAIN CONTENT ===
if analysis_type == "📄 Analisis File CSV":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
                <h2 style='color: white; font-weight: 600;'>📁 Upload & Analisis File CSV</h2>
                """, unsafe_allow_html=True)

    
    # Upload area
    st.markdown("""
    <div class="upload-area">
        <p>File harus memiliki kolom 'ulasan' yang berisi teks untuk dianalisis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV", 
        type="csv",
        help="File CSV harus memiliki kolom 'ulasan'"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'ulasan' not in df.columns:
                st.error("❌ File harus memiliki kolom 'ulasan'")
                st.info("Kolom yang ditemukan: " + ", ".join(df.columns.tolist()))
            else:
                # Preprocessing
                original_count = len(df)
                df = df.dropna(subset=['ulasan']).reset_index(drop=True)
                df['processed'] = df['ulasan'].apply(preprocess_text)
                
                # Prediction
                X = vectorizer.transform(df['processed'])
                df['sentimen'] = model.predict(X)
                df['sentimen'] = df['sentimen'].astype(str).str.lower().map({
                    'positive': 'Positif',
                    'negative': 'Negatif',
                    'neutral': 'Netral'
                })
                
                # Success message
                st.success(f"✅ Berhasil memproses {len(df)} ulasan dari {original_count} data")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                sentiment_counts = df['sentimen'].value_counts()
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(df)}</h3>
                        <p>Total Ulasan</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    positive_count = sentiment_counts.get('Positif', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{positive_count}</h3>
                        <p>Positif</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    negative_count = sentiment_counts.get('Negatif', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{negative_count}</h3>
                        <p>Negatif</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    neutral_count = sentiment_counts.get('Netral', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{neutral_count}</h3>
                        <p>Netral</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data preview
                st.markdown("### 📝 Preview Data")
                st.dataframe(
                    df[['ulasan', 'sentimen']].head(10),
                    use_container_width=True
                )
                
                # Visualization
                st.markdown("### 📊 Visualisasi Sentimen")
                
                if chart_type in ["Bar Chart", "Both"]:
                    fig_bar = px.bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        title="Distribusi Sentimen",
                        labels={'x': 'Sentimen', 'y': 'Jumlah'},
                        color=sentiment_counts.index,
                        color_discrete_map={
                            'Positif': '#70b6df',
                            'Negatif': '#49a4d8',
                            'Netral': '#115aad'
                        }
                    )
                    fig_bar.update_layout(
                        template="plotly_white",
                        title_font_size=20,
                        showlegend=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                if chart_type in ["Pie Chart", "Both"]:
                    fig_pie = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Persentase Sentimen",
                        hole=0.35, 
                        color=sentiment_counts.index,
                        
                        color_discrete_map={
                            'Positif': "#70b6df",   
                            'Negatif': "#49a4d8",   
                            'Netral':  "#115aad"    
                        }
                    )
                    fig_pie.update_layout(
                        template="plotly_white",
                        title_font=dict(size=20, color="#333333"),
                        margin=dict(t=60, b=40, l=20, r=20),
                        legend_title_text="Sentimen",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.3,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=12)
                        )
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # WordCloud
                if show_wordcloud:
                    st.markdown("### ☁️ Word Cloud")
                    
                    selected_sentiment = st.selectbox(
                        "Pilih sentimen untuk Word Cloud:",
                        sentiment_counts.index.tolist(),
                        key="wordcloud_sentiment"
                    )
                    
                    if selected_sentiment:
                        filtered_df = df[df['sentimen'] == selected_sentiment]
                        if not filtered_df.empty:
                            all_text = " ".join(filtered_df['ulasan'].astype(str))
                            
                            if all_text.strip():
                                wordcloud = generate_wordcloud(all_text)
                                
                                fig_wc, ax = plt.subplots(figsize=(12, 6))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                ax.set_title(f'Word Cloud - {selected_sentiment}', fontsize=16, fontweight='bold')
                                
                                st.pyplot(fig_wc)
                            else:
                                st.warning("Tidak ada teks yang cukup untuk membuat word cloud")
                        else:
                            st.warning("Tidak ada data untuk sentimen yang dipilih")
                
                # Download button
                st.markdown("### 💾 Download Hasil")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Analisis",
                    data=csv,
                    file_name=f"hasil_analisis_sentimen_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Klik untuk mendownload hasil analisis dalam format CSV"
                )
                
                # === Hasil Evaluasi Model ===
                st.markdown("### 📊 Hasil Evaluasi Model SVM")
                
                try:
                    with open('y_test.pkl', 'rb') as f:
                        y_test = pickle.load(f)
                    with open('y_pred_svm.pkl', 'rb') as f:
                        y_pred_svm = pickle.load(f)

                    accuracy_svm = accuracy_score(y_test, y_pred_svm)
                    accuracy_svm_percentage = accuracy_svm * 100
                    report = classification_report(y_test, y_pred_svm, output_dict=True)
                    conf_matrix = confusion_matrix(y_test, y_pred_svm)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(label="🎯 Akurasi SVM", value=f"{accuracy_svm_percentage:.2f}%")

                    with col2:
                        st.markdown("**Classification Report:**")
                        st.dataframe(pd.DataFrame(report).transpose())

                    st.markdown("**Confusion Matrix:**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
                    ax.set_xlabel('Predicted Labels')
                    ax.set_ylabel('True Labels')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)

                except Exception as e:
                    st.warning("File evaluasi model tidak ditemukan atau belum tersedia.")
                    st.info("Pastikan file 'y_test.pkl' dan 'y_pred_svm.pkl' ada di direktori yang sama.")

        except Exception as e:
            st.error(f"❌ Error saat memproses file: {str(e)}")
            st.info("Pastikan file CSV memiliki format yang benar dan kolom 'ulasan' ada.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif analysis_type == "✍️ Analisis Manual":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; font-weight: 600;'>✍️ Analisis Teks Manual</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: white;'>📝 Masukkan Teks</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: white;'>Ketik atau paste teks yang ingin dianalisis:</p>", unsafe_allow_html=True)
    user_input = st.text_area(
        label="", 
        height=150,
        placeholder="Masukan Teks",
        help="Masukkan teks dalam bahasa Indonesia untuk analisis sentimen"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_button = st.button("🔍 Analisis Sentimen", use_container_width=True)
    
    with col2:
        if st.button("Hapus Teks", use_container_width=True):
            st.rerun()
    
    if analyze_button:
        if not user_input.strip():
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Sedang menganalisis sentimen..."):
                # Preprocessing
                processed_text = preprocess_text(user_input)
                
                # Prediction
                X = vectorizer.transform([processed_text])
                prediction = model.predict(X)[0]
                
                # Show result
                st.markdown("### 🎯 Hasil Analisis")
                
                if prediction == "Positif":
                    st.markdown(f'<div class="result-positive">😊 Sentimen: {prediction}</div>', unsafe_allow_html=True)
                elif prediction == "Negatif":
                    st.markdown(f'<div class="result-negative">😞 Sentimen: {prediction}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-neutral">😐 Sentimen: {prediction}</div>', unsafe_allow_html=True)
                
                # Show preprocessing details
                with st.expander("🔧 Detail Preprocessing"):
                    st.markdown("**Teks Asli:**")
                    st.text(user_input)
                    st.markdown("**Teks Setelah Preprocessing:**")
                    st.text(processed_text if processed_text else "Tidak ada teks yang tersisa setelah preprocessing")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.9);">
    <p>🚀 Instagram Sentiment Analysis Dashboard | Powered by Machine Learning</p>
    <p>💡 Built with Streamlit, TF-IDF, and SVM</p>
</div>
""", unsafe_allow_html=True)
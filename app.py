import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load model CNN
model = load_model("cnn_model.h5")

# Label klasifikasi
classes = {
    0: ('AKIEC', 'Actinic Keratoses and Intraepithelial Carcinoma'),
    1: ('BCC', 'Basal Cell Carcinoma'),
    2: ('BKL', 'Benign Keratosis-like Lesions'),
    3: ('DF', 'Dermatofibroma'),
    4: ('NV', 'Melanocytic Nevi'),
    5: ('VASC', 'Vascular Lesions'),
    6: ('MEL', 'Melanoma')
}

# Konfigurasi halaman
st.set_page_config(page_title="Pendeteksi Dini Kanker Kulit", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    .styled-upload > div {
        border: 2px dashed #999 !important;
        padding: 2em;
        border-radius: 10px;
        background-color: #1e1e1e;
        color: #ccc;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 200px;
    }
    .styled-upload:hover > div {
        border-color: white !important;
        background-color: #2e2e2e;
        color: white;
    }
    .result-box {
        background-color: #d10000;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Judul
st.markdown("<h2 style='text-align: center;'>Pendeteksi Dini Kanker Kulit</h2>", unsafe_allow_html=True)

# Klasifikasi
st.markdown("#### 7 Klasifikasi Kanker Kulit:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- AKIEC (Actinic Keratoses and Intraepithelial Carcinoma)")
    st.markdown("- BCC (Basal Cell Carcinoma)")
    st.markdown("- BKL (Benign Keratosis-like Lesions)")
with col2:
    st.markdown("- DF (Dermatofibroma)")
    st.markdown("- MEL (Melanoma)")
    st.markdown("- NV (Melanocytic Nevi)")
    st.markdown("- VASC (Vascular Lesions)")

# Upload box styled
st.markdown("### Upload Gambar:")
with st.container():
    with st.expander("", expanded=True):
        st.markdown("""
        <div class="styled-upload">
            <div>
                <div style="font-size: 2em;">📤</div>
                <small>Drag & drop file di sini<br>atau klik untuk browse</small>
                <small>(Maks 200MB, JPG/JPEG/PNG)</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

# Tombol prediksi
if st.button("Prediksi Kanker Kulit"):
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

        # Preprocessing gambar
        img_resized = image.resize((28, 28))  # Sesuai input model
        img_array = np.array(img_resized) / 255.0  # Normalisasi
        img_array = img_array.reshape(1, 28, 28, 3)

        # Prediksi
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        class_name, description = classes[predicted_class]
        confidence = prediction[0][predicted_class] * 100  # Dalam persen

        # Tampilkan hasil
        # Tampilkan hasil dengan confidence level
        st.markdown(
            f"<div class='result-box'>"
            f"Prediksi: {class_name} - {description}<br>"
            f"Confidence: {confidence:.2f}%"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("#### Confidence Tiap Kelas:")
        for i, prob in enumerate(prediction[0]):
            label, full_name = classes[i]
            st.markdown(f"- {label} ({full_name}): **{prob * 100:.2f}%**")
        st.markdown(f"<div class='result-box'>Prediksi: {class_name} - {description}</div>", unsafe_allow_html=True)
    else:
        st.warning("Harap unggah gambar terlebih dahulu.")

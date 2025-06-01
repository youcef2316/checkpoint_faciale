import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Application de Détection de Visages")

st.write("""
Cette application détecte les visages dans les images téléchargées.  
Veuillez télécharger une image et ajuster les paramètres selon vos besoins.
""")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

min_neighbors = st.slider("Ajustez le paramètre minNeighbors :", 1, 10, 3)
scale_factor = st.slider("Ajustez le paramètre scaleFactor :", 1.01, 2.0, 1.3, step=0.01)
rectangle_color_hex = st.color_picker("Choisissez la couleur des rectangles :", "#FF0000")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))

    # Convert RGB to BGR pour OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Convertir la couleur hex en BGR tuple
    hex = rectangle_color_hex.lstrip('#')
    bgr_color = tuple(int(hex[i:i+2], 16) for i in (4, 2, 0))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), bgr_color, 2)

    # Convertir image BGR à RGB pour affichage Streamlit
    image_display = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    st.image(image_display, caption='Image avec visages détectés', use_column_width=True)

    # Bouton pour enregistrer et télécharger l'image
    if st.button("Enregistrer l'image"):
        is_success, buffer = cv2.imencode(".png", image_cv)
        if is_success:
            io_buf = io.BytesIO(buffer)
            st.download_button(
                label="Télécharger l'image détectée",
                data=io_buf,
                file_name="detected_faces.png",
                mime="image/png"
            )
        else:
            st.error("Erreur lors de la création du fichier à télécharger.")

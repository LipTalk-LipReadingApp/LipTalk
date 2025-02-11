import os
import tensorflow as tf
import cv2
import streamlit as st
import io
import imageio

from utils import load_video, num_to_char, load_alignment
from modelutil import load_model
from landmark_face_detection import detect_faces_and_mouths


from deep_translator import GoogleTranslator
from gtts import gTTS
import langcodes
from pathlib import Path

# Set the layout to the streamlit app as wide
st.set_page_config(layout="wide")

# Setup the sidebar
with st.sidebar:
    st.image("../Streamlit/Lip-reading-image.png")
    st.title("Lip Reading with AI")
    st.info("This application is developed as part of the final Data Science project!")

st.title("LipTalk App")

# Generate two columns
col1, col2 = st.columns(2)
uploaded_video = None
# Rendering the video
with col1:
  st.info("Choose your video")

  video = st.radio(
      "Select your video generation method",
      ["Upload a video", "Open live webcam"]
      )

  if video == "Upload a video":

    uploaded_video = st.file_uploader("Upload a video") 
    if uploaded_video is not None:

        vid = uploaded_video.name
        file_path = os.path.join("../Streamlit/Videos/", vid) 
        uploaded_video = open(file_path, "rb")
        video_bytes = uploaded_video.read()
        st.video(video_bytes)

  elif video == "Open live webcam":
    col5, col6 = st.columns(2)

    result_start = col5.button("Start", type="primary", key="start_button")
    result_stop = col6.button("Stop", key="stop_button")

    video_capture = None
    video_writer = None

    if result_start:
        # Open the camera
        video_capture = cv2.VideoCapture(0)
        
        # Get the default frame size from the camera (to ensure matching dimensions)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter("test_video.mp4", fourcc, 10, (frame_width, frame_height))
  
        # Create a Streamlit placeholder for the video frame
        video_placeholder = st.empty()

        while True:
            # Read a frame from the camera
            ret, frame = video_capture.read()

            if not ret:
                break

            # Perform face and mouth detection
            frame = detect_faces_and_mouths(frame)

            # Convert frame to RGB (as OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            video_placeholder.image(frame_rgb, channels="RGB")
            
            # Write the frame to the file
            video_writer.write(frame)     
         
    elif result_stop:
        # Release video capture and writer
        if video_capture is not None:
            video_capture.release()
        if video_writer is not None:
            video_writer.release()
    
    if video == "Open live webcam":
      file_path="../Streamlit/test_video.mp4"
      with open(file_path, "rb") as f:
        video_bytes = f.read()
      uploaded_video = io.BytesIO(video_bytes)

      # Get the base name of the file path (i.e., the file name without the directory part)
      vid = os.path.basename(file_path)   
    
with col2:

  st.info("Set the translation parameters")

  audio = st.radio(
      "Do you want to hear the audio of the video?",
      ["No", "Yes"]
      )

  if audio == "Yes":
    option_audio = st.selectbox(
    "Select audio language:",
    ("Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Catalan", "Cebuano", "Chichewa", "Chinese", "Corsican", "Croatian", "Czech", "Danish", "Dutch", "English", "Esperanto", "Estonian", "Filipino", "Finnish", "French", "Frisian", "Galician", "Georgian", "German", "Greek", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian", "Hebrew", "Hindi", "Hmong", "Hungarian", "Icelandic", "Igbo", "Indonesian", "Irish", "Italian", "Japanese", "Javanese", "Kannada", "Kazakh", "Khmer", "Kinyarwanda", "Korean", "Kurdish", "Kyrgyz", "Lao", "Latin", "Latvian", "Lithuanian", "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi", "Mongolian", "Myanmar", "Nepali", "Norwegian", "Odia", "Pashto", "Persian", "Polish", "Portuguese", "Punjabi", "Romanian", "Russian", "Samoan", "Scots Gaelic", "Serbian", "Sesotho", "Shona", "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali", "Spanish", "Sundanese", "Swahili", "Swedish", "Tajik", "Tamil", "Tatar", "Telugu", "Thai", "Turkish", "Turkmen", "Ukrainian", "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Xhosa", "Yiddish", "Yoruba", "Zulu"),
    index=None,
    placeholder="Select audio language...",
    )

  option_text = st.selectbox(
    "Select translation language:",
    ("Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque", "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Catalan", "Cebuano", "Chichewa", "Chinese", "Corsican", "Croatian", "Czech", "Danish", "Dutch", "English", "Esperanto", "Estonian", "Filipino", "Finnish", "French", "Frisian", "Galician", "Georgian", "German", "Greek", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian", "Hebrew", "Hindi", "Hmong", "Hungarian", "Icelandic", "Igbo", "Indonesian", "Irish", "Italian", "Japanese", "Javanese", "Kannada", "Kazakh", "Khmer", "Kinyarwanda", "Korean", "Kurdish", "Kyrgyz", "Lao", "Latin", "Latvian", "Lithuanian", "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi", "Mongolian", "Myanmar", "Nepali", "Norwegian", "Odia", "Pashto", "Persian", "Polish", "Portuguese", "Punjabi", "Romanian", "Russian", "Samoan", "Scots Gaelic", "Serbian", "Sesotho", "Shona", "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali", "Spanish", "Sundanese", "Swahili", "Swedish", "Tajik", "Tamil", "Tatar", "Telugu", "Thai", "Turkish", "Turkmen", "Ukrainian", "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Xhosa", "Yiddish", "Yoruba", "Zulu"),
    index=None,
    placeholder="Select translation language...",
  )

  if uploaded_video is not None and option_text is not None: 
    st.info("This is what the machine learning model sees when making a prediction")
    file_path = os.path.join("../Streamlit/Videos/", vid)
    video = load_video(tf.convert_to_tensor(file_path))
    imageio.mimsave("animation.gif", video, fps=10)
    st.image("animation.gif", width=400)

    model = load_model()
    yhat = model.predict(tf.expand_dims(video, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    
    #Create two columns for displaying the converted predictions and the real text
    col3, col4 = st.columns(2)

    with col3:
        st.info("Predicted Text")
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

    with col4:
        st.info("Real Text")
        real_text=load_alignment(file_path)
        st.text(real_text)

    st.info("This is the output translation")
    lang_code=langcodes.find(option_text).to_tag()
    translated_text = GoogleTranslator(source="auto", target=lang_code).translate(converted_prediction)
    st.text(translated_text)

    if audio == "Yes":
      st.write("Listen to the audio:")
      lang_code_audio=langcodes.find(option_audio).to_tag()
      translated_text_audio = GoogleTranslator(source="auto", target=lang_code_audio).translate(converted_prediction)
      speech = gTTS(text = translated_text_audio, lang=lang_code_audio, slow = False)
      speech.save("audio.mp3")
      st.audio("../Streamlit/audio.mp3", format="audio/mpeg", loop=False)

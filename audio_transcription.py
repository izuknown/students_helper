import streamlit as st
import whisper

st.title("Transcription Service")

#upload audio file with streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])


@st.cache
def load_whipser_model ():
    model = whisper.load_model("base") #there are various types of models available, each with differenct RAM requirments 
    return model

if st.sidebar.button("Load Whisper Model"):
    model = load_whipser_model()
    st.sidebar.success("Whisper Model Loaded")


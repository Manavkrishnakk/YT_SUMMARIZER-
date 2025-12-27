import streamlit as st
import os
import re
import torch
import yt_dlp
import whisper
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
from googletrans import Translator

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🎬 AI YouTube Video Summarizer", layout="wide")
st.title(" AI YouTube Video Summarizer")

# ------------------ HELPER: Extract Video ID ------------------
def extract_video_id(url_or_id):
    if len(url_or_id) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None

# ------------------ CACHED LOADERS ------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_translator():
    return Translator()

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)
    return model, device

# ------------------ STEP 1: Download Audio ------------------
def download_audio(video_url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        filename = ydl.prepare_filename(info)
        audio_file = filename.rsplit(".", 1)[0] + ".mp3"
    return audio_file

# ------------------ STEP 2: Transcribe if No Captions ------------------
def transcribe_to_english(audio_path):
    model, device = load_whisper_model()
    result = model.transcribe(audio_path, task="translate", fp16=(device == "cuda"))
    return result["text"]

# ------------------ STEP 3: Summarize Text ------------------
def summarize_text(text):
    summarizer = load_summarizer()
    max_chunk = 800
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ""
    for chunk in chunks:
        part = summarizer(chunk, max_length=70, min_length=30, do_sample=False)
        summary += part[0]["summary_text"] + " "
    return summary.strip()

# ------------------ STEP 4: Translate Summary ------------------
# ------------------ STEP 4: Translate Summary ------------------
def translate_summary(summary, target_language):
    translator = load_translator()
    if target_language.lower() == "english":
        return summary

    # Clean and chunk the text
    summary = summary.replace("\n", " ").strip()
    chunks = [summary[i:i+4000] for i in range(0, len(summary), 4000)]

    translated_parts = []
    for chunk in chunks:
        try:
            result = translator.translate(chunk, dest=target_language.lower())
            if result and hasattr(result, "text") and result.text:
                translated_parts.append(result.text)
        except Exception as e:
            # If translation fails for a chunk, keep original
            translated_parts.append(f"[Untranslated chunk due to error: {e}]")

    # Safely join parts into one string
    return " ".join(part for part in translated_parts if part)


# ------------------ LAYOUT ------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    video_input = st.text_input(
        "Enter YouTube URL or Video ID", 
        placeholder=""
    )
    target_language = st.selectbox(
        "Choose translation language",
        ["English", "Hindi", "Malayalam", "Tamil", "French", "Spanish", "German"]
    )
    summarize_button = st.button("Get Summary")

# ------------------ MAIN LOGIC ------------------
with right_col:
    if video_input and summarize_button:
        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("❌ Invalid YouTube URL or Video ID.")
            st.stop()
        video_url = f"https://www.youtube.com/embed/{video_id}"
        st.markdown(f"""<iframe width="400" height="225" src="{video_url}" frameborder="0" allowfullscreen></iframe>""", unsafe_allow_html=True)

        # Try fetching captions with your method
        try:
            with st.spinner("⏳ Fetching captions..."):
                ytt_api = YouTubeTranscriptApi()
                transcript_snippets = ytt_api.fetch(video_id)
                full_text = " ".join([snippet.text for snippet in transcript_snippets])
                st.success("✅ Captions fetched successfully!")
        except (TranscriptsDisabled, NoTranscriptFound, Exception):
            st.warning("⚠️ No captions found. Using Whisper to transcribe audio...")
            with st.spinner("🎧 Downloading audio and transcribing..."):
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                audio_path = download_audio(video_url)
                full_text = transcribe_to_english(audio_path)
                st.success("✅ Audio transcribed successfully using Whisper!")

        if not full_text.strip():
            st.error("❌ Unable to obtain transcript or audio text.")
            st.stop()

        # Show transcript
        with st.expander("📜 View Full Transcript"):
            st.text_area("Transcript", full_text, height=300)

        # Summarize
        with st.spinner("⚙️ Summarizing..."):
            summary_text = summarize_text(full_text)

        # Translate
        with st.spinner(f"Translating summary to {target_language}..."):
            translated_summary = translate_summary(summary_text, target_language)

        # Display
       # Display summary in bullet points
        st.subheader("📌 English Summary")
        for sentence in re.split(r'(?<=[.!?]) +', summary_text):
            if sentence.strip():  # ignore empty sentences
                st.markdown(f"- {sentence.strip()}")

        if target_language.lower() != "english":
            st.subheader(f"Translated Summary ({target_language})")
            for sentence in re.split(r'(?<=[.!?]) +', translated_summary):
                if sentence.strip():
                    st.markdown(f"- {sentence.strip()}")



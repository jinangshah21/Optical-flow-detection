import streamlit as st
import tempfile
import subprocess
import os
from cv_LK import process_video
from cv_HS import process_video2

# Convert AVI to MP4 for browser playback
def convert_to_mp4(input_file, output_file):
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Show processed videos for all 4 methods
def show_all_methods(input_path, prefix="result"):
    methods = {
        "Lucas-Kanade": "LK",
        "Horn-Schunck": "HS",
        "Farneback": "Farneback",
        "RLOF": "RLOF"  # Replace or rename based on actual methods
    }
    st.subheader(f"Input Video")
    st.video(input_path, format="video/mp4")
    st.info("Processing Lucas-Kanade method...")

    # for label, method in methods.items():
    raw_out = tempfile.NamedTemporaryFile(delete=False, suffix="_LK.avi").name
    mp4_out = tempfile.NamedTemporaryFile(delete=False, suffix="_LK.mp4").name
    process_video(input_path, raw_out)
    convert_to_mp4(raw_out, mp4_out)
    st.subheader(f"Lucas-Kanade Method")
    st.video(mp4_out, format="video/mp4")

    st.info("Processing Horn-Schunck method...")
    raw_out = tempfile.NamedTemporaryFile(delete=False, suffix="_HS.avi").name
    mp4_out = tempfile.NamedTemporaryFile(delete=False, suffix="_HS.mp4").name

    process_video2(input_path, raw_out)
    convert_to_mp4(raw_out, mp4_out)
    st.subheader(f"Horn-Schunck Method")
    st.video(mp4_out, format="video/mp4")

def show_all_sample_methods(input_path, prefix="result"):
    methods = {
        "Lucas-Kanade": "LK",
        "Horn-Schunck": "HS",
        "Farneback": "Farneback",
        "RLOF": "RLOF"  # Replace or rename based on actual methods
    }

    st.subheader(f"Sample Video")
    st.video(input_path, format="video/mp4")
    st.subheader(f"Lucas-Kanade Method")
    st.video("sample_output_LK.mp4", format="video/mp4")

    st.subheader(f"Horn-Schunck Method")
    st.video("sample_output_HS.mp4", format="video/mp4")
    st.subheader(f"RAFT Method")
    st.video("sample_output_RAFT.mp4", format="video/mp4")

# Main interface
st.title("🎥 Optical Flow Detection")

option = st.radio("Choose an option:", ("Use Sample Video", "Upload Your Video"))

if option == "Use Sample Video":
    if st.button("Run on Sample Video"):
        sample_input = "sample_input.mp4"  # Make sure this file is in your working dir
        show_all_sample_methods(sample_input, prefix="sample")

elif option == "Upload Your Video":
    video_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        current_dir = os.getcwd()
        # uploaded_path = os.path.join(current_dir, "uploaded_input.mp4")
        uploaded_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        with open(uploaded_path, "wb") as f:
            f.write(video_file.read())

        if st.button("Run on Uploaded Video"):
            show_all_methods(uploaded_path, prefix="user_uploaded")

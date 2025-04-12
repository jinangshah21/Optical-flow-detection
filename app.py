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
    st.subheader(f"Input Video")
    st.video(input_path, format="video/mp4")
    st.info("Processing Lucas-Kanade method...")

    # for label, method in methods.items():
    raw_out = os.path.join(os.getcwd(), "uploaded_video_output_LK.avi")
    mp4_out = os.path.join(os.getcwd(), "uploaded_video_output_LK.mp4")
    process_video(input_path, raw_out)
    convert_to_mp4(raw_out, mp4_out)
    st.subheader(f"Lucas-Kanade Method")
    st.video(mp4_out, format="video/mp4")

    st.info("Processing Horn-Schunck method...")
    raw_out = os.path.join(os.getcwd(), "uploaded_video_output_HS.avi")
    mp4_out = os.path.join(os.getcwd(), "uploaded_video_output_HS.mp4")

    process_video2(input_path, raw_out)
    convert_to_mp4(raw_out, mp4_out)
    st.subheader(f"Horn-Schunck Method")
    st.video(mp4_out, format="video/mp4")

def show_all_sample_methods(input_path, s_no, prefix="result"):

    st.subheader(f"Sample Video")
    st.video(input_path, format="video/mp4")
    st.subheader(f"Lucas-Kanade Method")
    st.video(f"sample_output{s_no}_LK.mp4", format="video/mp4")

    st.subheader(f"Horn-Schunck Method")
    st.video(f"sample_output{s_no}_HS.mp4", format="video/mp4")
    st.subheader(f"RAFT Method")
    st.video(f"sample_output{s_no}_RAFT.mp4", format="video/mp4")
    st.subheader(f"Farneback Method")
    st.video(f"sample_output{s_no}_Farneback.mp4", format="video/mp4")

# Main interface
st.title("🎥 Optical Flow Detection")

option = st.radio("Choose an option:", ("Use Sample Video 1", "Use Sample Video 2", "Upload Your Video"))

if option == "Use Sample Video 1":
    if st.button("Run on Sample Video"):
        sample_input = "sample_input.mp4"  # Make sure this file is in your working dir
        show_all_sample_methods(sample_input, "1", prefix="sample")

if option == "Use Sample Video 2":
    if st.button("Run on Sample Video"):
        sample_input = "sample_input2.mp4"  # Make sure this file is in your working dir
        show_all_sample_methods(sample_input, "2", prefix="sample")

elif option == "Upload Your Video":
    video_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        current_dir = os.getcwd()
        # uploaded_path = os.path.join(current_dir, "uploaded_input.mp4")
        uploaded_path = os.path.join(os.getcwd(), "uploaded_video.mp4")

        with open(uploaded_path, "wb") as f:
            f.write(video_file.read())

        if st.button("Run on Uploaded Video"):
            show_all_methods(uploaded_path, prefix="user_uploaded")

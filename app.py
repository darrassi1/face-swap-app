import os
import logging
from tempfile import NamedTemporaryFile
import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import requests
import onnxruntime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Initialize global variables
app = None
swapper = None

# Set Streamlit page config
st.set_page_config(page_title="FaceSwap App by Younes Darrassi")

def download_model():
    """
    Download the face-swapping model if it doesn't exist locally.
    """
    url = "https://cdn.adikhanofficial.com/python/insightface/models/inswapper_128.onnx"
    filename = url.split('/')[-1]
    filepath = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(filepath):
        st.info(f"Downloading {filename}... This may take a few minutes.")
        try:
            response = requests.get(url, stream=True, timeout=180)
            response.raise_for_status()
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logging.info(f"{filename} downloaded successfully.")
            st.success(f"{filename} downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            st.error(f"Failed to download model: {e}")
            raise
    else:
        logging.info(f"{filename} already exists in the directory.")
        st.info(f"{filename} already exists in the directory.")

def swap_faces(target_image, target_face, source_face):
    """
    Swap faces in the target image using the source face.
    """
    try:
        return swapper.get(target_image, target_face, source_face, paste_back=True)
    except Exception as e:
        logging.error(f"Error during face swapping: {e}")
        st.error(f"Error during face swapping: {e}")
        return None

def image_faceswap_app():
    """
    Streamlit app for swapping faces in images.
    """
    st.title("Face Swapper for Image")
    source_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

    if source_image and target_image:
        with st.spinner("Swapping... Please wait."):
            try:
                # Read and process images
                source_image = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)
                target_image = cv2.imdecode(np.frombuffer(target_image.read(), np.uint8), -1)
                source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

                # Detect faces
                source_faces = app.get(source_image)
                if not source_faces:
                    raise ValueError("No faces found in the source image.")
                source_face = sorted(source_faces, key=lambda x: x.bbox[0])[0]

                target_faces = app.get(target_image)
                if not target_faces:
                    raise ValueError("No faces found in the target image.")
                target_face = sorted(target_faces, key=lambda x: x.bbox[0])[0]

                # Swap faces
                swapped_image = swap_faces(target_image, target_face, source_face)
                if swapped_image is None:
                    raise ValueError("Face swapping failed.")

                # Display results
                st.success("Swapped Successfully!")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(source_image, caption="Source Image", use_column_width=True)
                with col2:
                    st.image(target_image, caption="Target Image", use_column_width=True)
                with col3:
                    st.image(swapped_image, caption="Swapped Image", use_column_width=True)
            except Exception as e:
                logging.error(f"Error during image processing: {e}")
                st.error(f"Error during image processing: {e}")

def video_faceswap_app():
    """
    Streamlit app for swapping faces in videos.
    """
    st.title("Face Swapper for Video")
    source_image = st.file_uploader("Upload Source Face Image", type=["jpg", "jpeg", "png"])
    target_video = st.file_uploader("Upload Target Video", type=["mp4"])

    if source_image and target_video:
        with st.spinner("Processing... This may take a while."):
            try:
                # Read source image
                source_image = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)

                # Save target video to a temporary file
                temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video.write(target_video.read())
                output_video_path = os.path.splitext(temp_video.name)[0] + '_output.mp4'

                # Process video
                process_video(source_image, temp_video.name, output_video_path)

                # Display result
                st.success("Processing complete!")
                st.subheader("Your video is ready:")
                st.video(output_video_path)
            except Exception as e:
                logging.error(f"Error during video processing: {e}")
                st.error(f"Error during video processing: {e}")

def process_video(source_img, video_path, output_video_path):
    """
    Process a video to swap faces frame by frame.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Detect source face
        source_faces = app.get(source_img)
        if not source_faces:
            raise ValueError("No faces found in the source image.")
        source_face = sorted(source_faces, key=lambda x: x.bbox[0])[0]

        # Process video frames
        progress_placeholder = st.empty()
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and swap faces in the frame
            target_faces = app.get(frame)
            if target_faces:
                frame = swap_faces(frame, target_faces[0], source_face)
            out.write(frame)

            # Update progress
            frame_count += 1
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time if elapsed_time > 0 else 0
            remaining_time_seconds = max(0, (total_frames - frame_count) / frames_per_second) if frames_per_second > 0 else 0
            remaining_minutes, remaining_seconds = divmod(remaining_time_seconds, 60)
            elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)

            progress_placeholder.text(
                f"Processed Frames: {frame_count}/{total_frames} | "
                f"Elapsed Time: {int(elapsed_minutes)}m {int(elapsed_seconds)}s | "
                f"Remaining Time: {int(remaining_minutes)}m {int(remaining_seconds)}s"
            )

        cap.release()
        out.release()
        logging.info("Video processing completed successfully.")
    except Exception as e:
        logging.error(f"Error during video processing: {e}")
        st.error(f"Error during video processing: {e}")

def main():
    """
    Main function to run the Streamlit app.
    """
    global app, swapper

    try:
        # Configure ONNX runtime providers
        providers = ['CPUExecutionProvider']
        logging.info(f"Using providers: {providers}")

        # Initialize models
        if app is None:
            app = FaceAnalysis(name='buffalo_l')
            app.prepare(ctx_id=0, det_size=(640, 640), providers=providers)
            logging.info("Face Analysis model initialized")

        if swapper is None:
            download_model()  # Download model if not available
            swapper = insightface.model_zoo.get_model(
                'inswapper_128.onnx',
                root=os.path.dirname(__file__),
                providers=providers
            )
            logging.info("Face swapper model loaded")

        # App selection
        app_selection = st.sidebar.radio("Select App", ("Image Face Swapping", "Video Face Swapping"))
        if app_selection == "Image Face Swapping":
            image_faceswap_app()
        elif app_selection == "Video Face Swapping":
            video_faceswap_app()

    except Exception as e:
        logging.error(f"Critical error in main function: {e}")
        st.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()
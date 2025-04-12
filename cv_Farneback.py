import os
import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

def process_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame")
        return
    
    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Create mask for flow visualization
    mask = np.zeros_like(prev_frame)
    
    # Farneback parameters for optimal results
    flow_params = dict(
        pyr_scale=0.5,     # Pyramid scale between layers
        levels=5,          # Number of pyramid layers
        winsize=15,        # Averaging window size
        iterations=3,      # Number of iterations at each pyramid level
        poly_n=5,          # Size of pixel neighborhood
        poly_sigma=1.2,    # Standard deviation for polynomial expansion
        flags=0            # Flags (can use cv2.OPTFLOW_FARNEBACK_GAUSSIAN for smoother results)
    )
    
    print(f"Processing video with {total_frames} frames...")
    frame_count = 0
    
    # Process each frame
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
        
        # Convert current frame to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **flow_params)
        
        # Create color-coded flow visualization
        flow_color = flow_to_color(flow)
        
        # Blend original frame with flow color
        alpha = 0.7  # Original frame weight
        blended = cv2.addWeighted(curr_frame, alpha, flow_color, 1-alpha, 0)
        
        # Draw flow arrows on top
        flow_viz = draw_flow_arrows(blended, flow)
        
        # Add frame counter
        cv2.putText(flow_viz, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(flow_viz)
        
        # Update previous frame
        prev_gray = curr_gray
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to {output_video_path}")

# Draw flow arrows on the frame
def draw_flow_arrows(frame, flow, step=16, scale=1.5, color=(0, 255, 0)):
    h, w = flow.shape[:2]
    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    
    fx, fy = flow[y_coords, x_coords].T
    
    # Create mask of valid flow (with minimum magnitude)
    magnitudes = np.sqrt(fx*fx + fy*fy)
    threshold = 1.0  # Minimum flow magnitude to display
    valid_mask = magnitudes > threshold
    
    lines = np.vstack([x_coords[valid_mask], y_coords[valid_mask],
                       (x_coords + fx * scale)[valid_mask],
                       (y_coords + fy * scale)[valid_mask]]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    # Draw the flow arrows
    vis = frame.copy()
    arrow_thickness = 2
    for (x1, y1), (x2, y2) in lines:
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, arrow_thickness, tipLength=0.3)
    
    return vis

# Flow to color visualization function
def flow_to_color(flow):
    # Convert flow to polar coordinates (magnitude, angle)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV color representation
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    # Normalize magnitude for better visualization
    # Apply log normalization to better visualize small and large movements
    mag_max = np.max(mag)
    if mag_max > 0:
        mag = np.log1p(mag)  # log(1+x) to avoid log(0)
        mag = np.clip(mag * 255 / np.max(mag), 0, 255).astype(np.uint8)
    else:
        mag = np.zeros_like(mag)
    
    # Map angle to hue
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue (0-180) based on angle
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = mag  # Value based on magnitude
    
    # Convert HSV to BGR for display
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr

def process_video4(input_video, output_video):
    print("Starting optical flow estimation process...")
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found.")
        return
    
    # Check if CUDA is available (just for info)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    try:
        # Process video
        process_video(input_video, output_video)
        
        print(f"Optical flow visualization complete. Output saved to: {output_video}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


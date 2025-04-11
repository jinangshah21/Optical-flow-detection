import torch
import torchvision
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import os
import subprocess

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Pretrained RAFT-Small Model ===
weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights).to(device).eval()

# === Preprocessing Function using Torchvision Weights ===
def preprocess(img1, img2):
    transform = weights.transforms()
    # Output will be tensors of shape [1, 3, H, W]
    return transform(img1, img2)

# === Draw Arrows on Optical Flow Field ===
def draw_arrows(frame, flow, step=12, threshold=0.5, scale=1.0):
    """
    Draw arrows on optical flow field with improved visibility

    Args:
        frame: Original frame
        flow: Optical flow field [2, H, W]
        step: Grid step size
        threshold: Minimum flow magnitude to draw arrow
        scale: Scale factor for arrow length
    """
    h, w = flow.shape[1], flow.shape[2]

    # Calculate flow magnitude
    mag, ang = cv2.cartToPolar(flow[0], flow[1])

    # Create copy of frame for visualization
    vis = frame.copy()

    # Draw arrows on significant flows only
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[0, y, x], flow[1, y, x]
            magnitude = mag[y, x]

            # Only draw arrows for flows above threshold
            if magnitude > threshold:
                # Scale arrow length for better visibility
                end_x = int(x + dx * scale)
                end_y = int(y + dy * scale)

                # Keep end points within frame bounds
                end_x = max(0, min(end_x, w-1))
                end_y = max(0, min(end_y, h-1))

                # Draw arrow with color based on direction
                # Red-ish for horizontal, Blue-ish for vertical
                color = (
                    min(255, int(magnitude * 5)),  # B
                    0,                             # G
                    min(255, int(magnitude * 5))   # R
                )

                cv2.arrowedLine(vis,
                               (x, y),
                               (end_x, end_y),
                               color=color,
                               thickness=1,
                               tipLength=0.3)

    return vis

def process_video3(input_path, output_path):
    # === Load Input Video ===
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # === Output Video Writer ===
    out_arrows = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*'XVID'),
                        fps,
                        (width, height))

    # out_color = cv2.VideoWriter('output_flow_color.mp4',
    #                      cv2.VideoWriter_fourcc(*'mp4v'),
    #                      fps,
    #                      (width, height))

    # === Read First Frame ===
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read video.")
        cap.release()
        exit()

    # === Process Parameters ===
    # Try keeping resolution higher for better detection
    resize_dim = (640, 360)  
    flow_threshold = 0.8  # Minimum flow to show an arrow
    arrow_scale = 3.0  # Scale arrow length for visibility
    arrow_step = 20  # Spacing between arrows

    frame_count = 0

    # === Optical Flow Processing Loop ===
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}", end="\r")

        # Resize both frames
        frame1 = cv2.resize(prev_frame, resize_dim)
        frame2 = cv2.resize(next_frame, resize_dim)

        # Convert BGR to RGB and then to PIL
        img1_pil = to_pil_image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        img2_pil = to_pil_image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        # Apply preprocessing (includes normalization & batching)
        img1_t, img2_t = preprocess(img1_pil, img2_pil)

        # Ensure shape is [B, 3, H, W]
        if img1_t.dim() == 3:
            img1_t = img1_t.unsqueeze(0)
        if img2_t.dim() == 3:
            img2_t = img2_t.unsqueeze(0)

        img1_t, img2_t = img1_t.to(device), img2_t.to(device)

        # Run model inference
        with torch.no_grad():
            flow_predictions = model(img1_t, img2_t)
            # RAFT returns predictions at multiple iterations - use final one
            flow_up = flow_predictions[-1]  # Use the final prediction

        # Get flow as numpy array: [2, H, W]
        flow = flow_up[0].cpu().numpy()

        # Generate visualization
        vis_arrows = draw_arrows(frame1.copy(), flow,
                            step=arrow_step,
                            threshold=flow_threshold,
                            scale=arrow_scale)

        # Resize back to original video resolution
        vis_arrows_resized = cv2.resize(vis_arrows, (width, height))

        # Write to output videos
        out_arrows.write(vis_arrows_resized)
        # out_color.write(vis_color_resized)

        # Display results in Colab (alternating between frames to save space)
        # if frame_count % 10 == 0:
        #     plt.figure(figsize=(16, 8))
        #     plt.subplot(121)
        #     plt.imshow(cv2.cvtColor(vis_arrows_resized, cv2.COLOR_BGR2RGB))
        #     plt.title("Flow Arrows")
        #     plt.axis('off')

        #     plt.subplot(122)
        #     plt.imshow(cv2.cvtColor(vis_color_resized, cv2.COLOR_BGR2RGB))
        #     plt.title("Flow Color Map")
        #     plt.axis('off')

        #     plt.tight_layout()
        #     plt.show()

        prev_frame = next_frame  # Move to next frame

    # === Cleanup ===
    print("\nProcessing complete!")
    cap.release()
    out_arrows.release()
    cv2.destroyAllWindows()

def convert_to_mp4(input_file, output_file):
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


raw_out = os.path.join(os.getcwd(), "samplex.avi")
mp4_out = os.path.join(os.getcwd(), "sample_output_RAFT.mp4")
input_path = os.path.join(os.getcwd(), "sample_input.mp4")

process_video3(input_path, raw_out)
convert_to_mp4(raw_out, mp4_out)
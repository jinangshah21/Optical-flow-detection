import cv2
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import supervision as sv
from scipy.ndimage import gaussian_filter
from numba import njit
import os
import subprocess

def get_grad(image, direction):
    if direction == 'x':
        return cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    elif direction == 'y':
        return cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    else:
        raise ValueError("Invalid direction. Use 'x' or 'y'.")

def preprocess(U, V):
    """
    Detects and corrects anomalies in U and V using the IQR method.
    Values outside the IQR range are clipped to reduce their effect.
    """
    def clip_to_iqr(arr):
        flat = arr.flatten()
        lower_bound = np.percentile(flat, 10)
        upper_bound = np.percentile(flat, 90)
        clipped = np.clip(arr, 0, upper_bound)
        return clipped

    speeds = np.sqrt(U**2 + V**2)
    new_speeds = clip_to_iqr(speeds)

    U,V = new_speeds/(speeds+1e-9) * U, new_speeds/(speeds+1e-9) * V
    return U,V

@njit
def get_neighbour_avg_numba(U):
    H, W = U.shape
    Ubar = np.zeros_like(U)
    for i in range(1, H-1):
        for j in range(1, W-1):
            Ubar[i, j] = (
                (U[i-1, j-1] + U[i-1, j+1] + U[i+1, j-1] + U[i+1, j+1]) / 12 +
                (U[i-1, j] + U[i, j-1] + U[i, j+1] + U[i+1, j]) / 6
            )
    return Ubar


@njit
def horn_schunck_solver(I_x, I_y, I_t, lamda, max_iter, tol):
    H, W = I_x.shape
    U = np.zeros((H, W))
    V = np.zeros((H, W))
    denom = lamda + I_x**2 + I_y**2

    loss1 = loss2 = 1e10
    for itr in range(max_iter):
        Ubar = get_neighbour_avg_numba(U)
        Vbar = get_neighbour_avg_numba(V)

        num = I_x * Ubar + I_y * Vbar + I_t

        Unew = Ubar - (num / denom) * I_x
        Vnew = Vbar - (num / denom) * I_y

        new_loss1 = np.sum(np.abs(U - Unew))
        new_loss2 = np.sum(np.abs(V - Vnew))

        if np.abs(new_loss1 - loss1) < tol and np.abs(new_loss2 - loss2) < tol:
            break

        U = Unew
        V = Vnew
        loss1, loss2 = new_loss1, new_loss2

    return U, V


def get_optical_flow_HS(frame1, frame2, lamda=5, max_iter=300, tol=1e-6, smoothening=0):
    frame1 = gaussian_filter(frame1, sigma=smoothening)
    frame2 = gaussian_filter(frame2, sigma=smoothening)
    
    I_x = get_grad(frame1, 'x')
    I_y = get_grad(frame1, 'y')
    I_t = frame2 - frame1

    # Call the JIT-compiled solver
    U, V = horn_schunck_solver(I_x, I_y, I_t, lamda, max_iter, tol)

    U, V = preprocess(U, V)
    return U, V

def draw_optical_flow(frame, U, V, step=20, color=(255, 0, 0), scale=1000.0, keep_ratio=0.5, only_corners=True, corner_conf=0.2):
    if only_corners:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=corner_conf, minDistance=5)
    
        # Ensure frame is in uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    
        # Convert grayscale to BGR
        if len(frame.shape) == 2:
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            annotated_frame = frame.copy()
    
        # Compute speed magnitude
        speed_magnitude = np.sqrt(U**2 + V**2)
        max_speed = np.max(speed_magnitude)
    
        if p0 is not None:
            for pt in p0:
                x, y = pt.ravel().astype(int)
                if 0 <= y < U.shape[0] and 0 <= x < U.shape[1]:
                    dx = U[y, x] * scale
                    dy = V[y, x] * scale
                    if np.sqrt(dx**2 + dy**2) > keep_ratio * max_speed:
                        pt1 = (x, y)
                        pt2 = (int(x + dx), int(y + dy))
                        cv2.arrowedLine(annotated_frame, pt1, pt2, color, thickness=1, tipLength=0.3)
    
        return annotated_frame
    else:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255)
            frame = frame.astype(np.uint8)    
    
        # Convert grayscale to BGR
        if len(frame.shape) == 2:
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            annotated_frame = frame.copy()
        
        h, w = U.shape
        speed_magnitude = np.sqrt(U**2 + V**2)
        less_speed = np.percentile(speed_magnitude, 60)
        max_speed = np.max(speed_magnitude)
        for y in range(0, h, step):
            for x in range(0, w, step):
                dx = np.mean(U[y:y+step, x:x+step]) * scale
                dy = np.mean(V[y:y+step, x:x+step]) * scale
                pt1 = (int(x), int(y))
                pt2 = (int(x + dx), int(y + dy))
                
                if speed_magnitude[y,x]>keep_ratio*max_speed:
                    cv2.arrowedLine(annotated_frame, pt1, pt2, color, thickness=1, tipLength=0.3)
        # plt.imshow(annotated_frame)
        # plt.show()
        return annotated_frame

def get_optical_flow_video(video_path, target_video_path, step=20, scale=800.0, keep_ratio=0.5, lamda=5, only_corners=True, corner_conf=0.2, resize_shape=(500, 334)):
    # Get video metadata
    video_info = sv.VideoInfo.from_video_path(video_path)
    video_info.width, video_info.height = resize_shape
    print(video_info)
    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    sink = sv.VideoSink(target_path=target_video_path, video_info=video_info)
    prev_gray = None
    with sink:
        for i, frame in enumerate(frame_generator):
            if i == 90: 
                break
            # Convert to grayscale
            gray = np.array(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), resize_shape)).astype(np.float32)
            if prev_gray is not None:
                U, V = get_optical_flow_HS(prev_gray, gray, lamda=lamda)
                annotated_frame = draw_optical_flow(cv2.resize(frame,resize_shape), U, V, keep_ratio=keep_ratio, step=step, scale=scale, only_corners=only_corners, corner_conf=corner_conf)
                sink.write_frame(annotated_frame.astype(np.uint8))
                print(f"Done for frame {i-1}")
            prev_gray = gray

def convert_to_mp4(input_file, output_file):
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# video_path = 'sample_input.mp4'
# target_video_path = "Optical_flow_HS.avi"
# target_video_path_fin = "Optical_flow_HS.mp4"
# video_path = os.path.join(os.getcwd(), 'sample_input.mp4')
# target_video_path = os.path.join(os.getcwd(), 'Optical_flow_HS.avi')
# target_video_path_fin = os.path.join(os.getcwd(), 'Optical_flow_HS.mp4')

def process_video2(video_path, target_video_path): 
    get_optical_flow_video(video_path, target_video_path, step=12, scale=500.0, keep_ratio=0.4, lamda=10000, only_corners=True, corner_conf=0.2)

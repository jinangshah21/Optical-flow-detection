import cv2
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import supervision as sv

from skimage.util import view_as_windows

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

def get_optical_flow_LK(frame1, frame2, patch_size=7, threshold=1e-5, smoothening=1):
    
    frame1 = gaussian_filter(frame1, sigma=smoothening)
    frame2 = gaussian_filter(frame2, sigma=smoothening)
    
    I_x = get_grad(frame1, 'x')
    I_y = get_grad(frame1, 'y')
    I_t = frame2 - frame1

    spread = patch_size // 2

    # Extract patches
    Ix_patches = view_as_windows(I_x, (patch_size, patch_size))  # shape: (H, W, p, p)
    Iy_patches = view_as_windows(I_y, (patch_size, patch_size))
    It_patches = view_as_windows(I_t, (patch_size, patch_size))

    H, W = Ix_patches.shape[:2]
    N = H * W

    # Reshape to (N, patch_size * patch_size)
    Ix_flat = Ix_patches.reshape(N, -1)
    Iy_flat = Iy_patches.reshape(N, -1)
    It_flat = It_patches.reshape(N, -1)

    # Compute elements of A matrix for all patches
    A11 = np.sum(Ix_flat * Ix_flat, axis=1)
    A12 = np.sum(Ix_flat * Iy_flat, axis=1)
    A22 = np.sum(Iy_flat * Iy_flat, axis=1)

    # Stack A matrices as (N, 2, 2)
    A = np.stack([
        np.stack([A11, A12], axis=1),
        np.stack([A12, A22], axis=1)
    ], axis=1)

    # Compute B vectors (N, 2)
    B1 = -np.sum(Ix_flat * It_flat, axis=1)
    B2 = -np.sum(Iy_flat * It_flat, axis=1)
    B = np.stack([B1, B2], axis=1)

    # Compute eigenvalues to check conditioning
    eigvals = np.linalg.eigvalsh(A)
    cond_mask = np.min(eigvals, axis=1) > threshold

    # Use np.linalg.solve for well-conditioned A, else pseudoinverse
    flow = np.zeros_like(B)
    well = cond_mask
    ill = ~cond_mask

    flow[well] = np.linalg.solve(A[well], B[well][:, :, None])[:, :, 0]
    A_inv = np.linalg.pinv(A[ill])
    flow[ill] = np.einsum('nij,nj->ni', A_inv, B[ill])

    # Create U, V maps
    U = np.zeros_like(frame1)
    V = np.zeros_like(frame1)
    U[spread:H + spread, spread:W + spread] = flow[:, 0].reshape(H, W)
    V[spread:H + spread, spread:W + spread] = flow[:, 1].reshape(H, W)

    # Visualization (optional)
    # fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    # axs[0, 0].imshow(frame1, cmap='gray')
    # axs[0, 0].set_title("Current Frame")
    # axs[0, 1].imshow(frame2, cmap='gray')
    # axs[0, 1].set_title("Next Frame")
    # axs[0, 2].imshow(I_t, cmap='gray')
    # axs[0, 2].set_title("I_t")
    # axs[1, 0].imshow(I_x, cmap='gray')
    # axs[1, 0].set_title("I_x")
    # axs[1, 1].imshow(I_y, cmap='gray')
    # axs[1, 1].set_title("I_y")
    # axs[1, 2].imshow(np.sqrt(U ** 2 + V ** 2), cmap='gray')
    # axs[1, 2].set_title("Flow Magnitude")
    # plt.tight_layout()
    # plt.show()

    U,V = preprocess(U,V)
    
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

from concurrent.futures import ThreadPoolExecutor

def get_optical_flow_video(
    video_path,
    target_video_path,
    resize_shape=(640, 480),  # (width, height)
    patch_size=7,
    step=14,
    scale=1000.0,
    keep_ratio=0.7,
    batch_size=6,
    max_workers=6,
    smoothening = 0,
    only_corners=True,
    corner_conf=0.2,
):
    def process_frame_pair(prev_gray, curr_gray, raw_frame):
        U, V = get_optical_flow_LK(prev_gray, curr_gray, patch_size=patch_size, smoothening=smoothening)
        return draw_optical_flow(raw_frame, U, V, step=step, scale=scale, keep_ratio=keep_ratio, only_corners=only_corners, corner_conf=corner_conf).astype(np.uint8)

    def generate_batches(generator, batch_size):
        batch = []
        for frame in generator:
            resized = cv2.resize(frame, resize_shape)  # Resize before batching
            batch.append(resized)
            if len(batch) == batch_size + 1:
                yield batch
                batch = batch[-1:]
        if len(batch) > 1:
            yield batch

    video_info = sv.VideoInfo.from_video_path(video_path)
    # Replace video_info dimensions with resize shape
    video_info.width, video_info.height = resize_shape

    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    sink = sv.VideoSink(target_path=target_video_path, video_info=video_info)

    with sink:
        for batch_index, batch in enumerate(generate_batches(frame_generator, batch_size)):
            if batch_index == 12:
                break
            batch_np = np.stack(batch)  # (batch+1, H, W, 3)
            gray_batch = batch_np.mean(axis=3).astype(np.float64)
            raw_batch = batch_np[:-1]

            input_triples = [
                (gray_batch[i], gray_batch[i + 1], raw_batch[i])
                for i in range(len(gray_batch) - 1)
            ]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(lambda args: process_frame_pair(*args), input_triples))

            for i, annotated in enumerate(results):
                sink.write_frame(annotated)
                print(f"Done for frame {batch_index * batch_size + i}")

def process_video(video_path, target_video_path):
    # video_path = 'input_v.mp4'
    # target_video_path = "Optical_flow_LK.mp4"
    get_optical_flow_video(video_path, target_video_path, step=15, scale=1000.0, keep_ratio=0.4, patch_size=11, only_corners=True, corner_conf=0.2)
    return target_video_path
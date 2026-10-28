import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from typing import List, Tuple


class ImagePreprocessor:
    def __init__(self):
        self.image_size = None

    # ------------------------------------------------------
    # 1. Load images from a folder
    # ------------------------------------------------------
    def load_images(self, folder: str, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Loads all images from a folder and resizes them.

        Input:  Folder path
        Output: NumPy array of shape (n_samples, H, W, C)
        """
        self.image_size = target_size
        images = []
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path)  # BGR by default
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)
        return np.array(images)

    # ------------------------------------------------------
    # 2. Convert to grayscale
    # ------------------------------------------------------
    def to_grayscale(self, images: np.ndarray) -> np.ndarray:
        return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])

    # ------------------------------------------------------
    # 3. Normalize pixel values
    # ------------------------------------------------------
    def normalize(self, images: np.ndarray, scale: float = 255.0) -> np.ndarray:
        return images.astype("float32") / scale

    # ------------------------------------------------------
    # 4. Noise reduction (Gaussian blur)
    # ------------------------------------------------------
    def denoise(self, images: np.ndarray, kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
        return np.array([cv2.GaussianBlur(img, kernel_size, 0) for img in images])

    # ------------------------------------------------------
    # 5. Data augmentation (flip + rotation)
    # ------------------------------------------------------
    def augment(self, images: np.ndarray, flip: bool = True, rotate: bool = True) -> np.ndarray:
        augmented = []
        for img in images:
            if flip:
                augmented.append(cv2.flip(img, 1))  # Horizontal flip
            if rotate:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)  # 15° rotation
                augmented.append(cv2.warpAffine(img, M, (w, h)))
        return np.concatenate([images, np.array(augmented)])

    # ------------------------------------------------------
    # 6. Flatten images (for ML models that require vectors)
    # ------------------------------------------------------
    def flatten(self, images: np.ndarray) -> np.ndarray:
        n_samples = images.shape[0]
        return images.reshape(n_samples, -1)

    # ------------------------------------------------------
    # 7. Train-test split
    # ------------------------------------------------------
    def split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)





class VideoPreprocessor:
    def __init__(self):
        self.frame_size = None
        self.max_frames = None

    # ------------------------------------------------------
    # 1. Load video and extract frames
    # ------------------------------------------------------
    def load_video(self, path: str, target_size: Tuple[int, int] = (128, 128), max_frames: int = 30) -> np.ndarray:
        """
        Loads a video, extracts frames, resizes them, and limits to max_frames.

        Input:  Video path
        Output: NumPy array of shape (frames, H, W, C)
        """
        self.frame_size = target_size
        self.max_frames = max_frames

        cap = cv2.VideoCapture(path)
        frames = []
        count = 0

        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
            count += 1
        cap.release()

        return np.array(frames)

    # ------------------------------------------------------
    # 2. Convert to grayscale
    # ------------------------------------------------------
    def to_grayscale(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        return [np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in video]) for video in videos]

    # ------------------------------------------------------
    # 3. Normalize pixel values
    # ------------------------------------------------------
    def normalize(self, videos: List[np.ndarray], scale: float = 255.0) -> List[np.ndarray]:
        return [video.astype("float32") / scale for video in videos]

    # ------------------------------------------------------
    # 4. Frame sampling (downsampling)
    # ------------------------------------------------------
    def sample_frames(self, video: np.ndarray, step: int = 2) -> np.ndarray:
        """
        Samples every Nth frame to reduce sequence length.
        """
        return video[::step]

    # ------------------------------------------------------
    # 5. Noise reduction
    # ------------------------------------------------------
    def denoise(self, videos: List[np.ndarray], kernel_size: Tuple[int, int] = (3, 3)) -> List[np.ndarray]:
        return [np.array([cv2.GaussianBlur(frame, kernel_size, 0) for frame in video]) for video in videos]

    # ------------------------------------------------------
    # 6. Data augmentation (flip + rotation on frames)
    # ------------------------------------------------------
    def augment(self, videos: List[np.ndarray], flip: bool = True, rotate: bool = True) -> List[np.ndarray]:
        augmented = []
        for video in videos:
            aug_frames = []
            for frame in video:
                new_frames = [frame]
                if flip:
                    new_frames.append(cv2.flip(frame, 1))  # Horizontal flip
                if rotate:
                    h, w = frame.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1)  # rotate 15°
                    new_frames.append(cv2.warpAffine(frame, M, (w, h)))
                aug_frames.extend(new_frames)
            augmented.append(np.array(aug_frames))
        return videos + augmented

    # ------------------------------------------------------
    # 7. Flatten frames (optional for classical ML)
    # ------------------------------------------------------
    def flatten(self, videos: List[np.ndarray]) -> np.ndarray:
        return np.array([video.reshape(video.shape[0], -1) for video in videos])

    # ------------------------------------------------------
    # 8. Train-test split
    # ------------------------------------------------------
    def split(self, X: List[np.ndarray], y: np.ndarray, test_size: float = 0.2):
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2


@dataclass
class IRSensorConfig:
    """Configuration of the synthetic IR sensor and scene."""
    width: int = 256
    height: int = 256
    min_temp: float = 15.0       # in °C, background cooler
    max_temp: float = 60.0       # in °C, hot objects
    bit_depth: int = 8           # 8-bit grayscale (0-255)
    noise_std: float = 2.0       # standard deviation of Gaussian noise in intensity domain
    blur_kernel_size: int = 5    # odd number
    atmosphere_attenuation: float = 0.95  # multiply intensities by this factor

    @property
    def max_intensity(self) -> int:
        return (2 ** self.bit_depth) - 1


class IRObject:
    """
    Represents a warm object inside the scene.
    For simplicity we support rectangles, circles, or blobs.
    """
    def __init__(
        self,
        initial_center: Tuple[int, int],
        size: Tuple[int, int],
        base_temp: float,
        temp_variation: float,
        shape: str = "rectangle"
    ):
        self.initial_center = np.array(initial_center, dtype=np.float32)
        self.size = size
        self.base_temp = base_temp
        self.temp_variation = temp_variation
        self.shape = shape

    def get_center_at_frame(self, frame_idx: int, total_frames: int) -> Tuple[int, int]:
        """
        Simple linear motion from left to right.
        You can customize this to move vertically/diagonally/etc.
        """
        # move horizontally across the frame
        progress = frame_idx / max(total_frames - 1, 1)
        dx = 80 * (progress - 0.5)  # move -40 to +40 pixels
        dy = 0                       # no vertical movement for now
        center = self.initial_center + np.array([dx, dy])
        return int(center[0]), int(center[1])

    def get_temperature_at_frame(self, frame_idx: int, total_frames: int) -> float:
        """
        Temperature oscillates slightly across frames (like heating/cooling).
        """
        phase = 2 * np.pi * frame_idx / max(total_frames, 1)
        delta = self.temp_variation * np.sin(phase)
        return self.base_temp + delta


class IRScene:
    """
    Handles creation of the temperature field: background + objects.
    """
    def __init__(self, config: IRSensorConfig, objects: List[IRObject]):
        self.config = config
        self.objects = objects

    def generate_background_temperature(self) -> np.ndarray:
        """
        Create a 2D temperature map for the background.
        We can simulate a soft gradient + small random variation.
        """
        h, w = self.config.height, self.config.width
        min_temp, max_temp = self.config.min_temp, self.config.min_temp + 5.0  # background range

        # Horizontal gradient
        x = np.linspace(0, 1, w)
        gradient = x[np.newaxis, :]  # shape (1, w)
        base_temp = min_temp + (max_temp - min_temp) * gradient  # shape (1, w)

        # Tile vertically
        temp_field = np.repeat(base_temp, h, axis=0)

        # Add small random variation
        noise = np.random.normal(loc=0.0, scale=0.3, size=(h, w))
        temp_field = temp_field + noise

        return temp_field

    def render_object_on_temperature(
        self,
        temp_field: np.ndarray,
        ir_object: IRObject,
        frame_idx: int,
        total_frames: int
    ) -> np.ndarray:
        """
        Draw the object into the temperature field.
        """
        h, w = temp_field.shape
        cx, cy = ir_object.get_center_at_frame(frame_idx, total_frames)
        obj_temp = ir_object.get_temperature_at_frame(frame_idx, total_frames)

        obj_w, obj_h = ir_object.size

        y_min = max(cy - obj_h // 2, 0)
        y_max = min(cy + obj_h // 2, h)
        x_min = max(cx - obj_w // 2, 0)
        x_max = min(cx + obj_w // 2, w)

        temp_copy = temp_field.copy()

        if ir_object.shape == "rectangle":
            temp_copy[y_min:y_max, x_min:x_max] = obj_temp
        elif ir_object.shape == "circle":
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (min(obj_w, obj_h) // 2) ** 2
            temp_copy[y_min:y_max, x_min:x_max][mask] = obj_temp
        else:
            # default rectangle if unknown shape
            temp_copy[y_min:y_max, x_min:x_max] = obj_temp

        # Optional: diffusion around warm area (simulate heat spreading)
        temp_copy = cv2.GaussianBlur(temp_copy.astype(np.float32), (3, 3), 0)

        return temp_copy

    def temperature_to_intensity(self, temp_field: np.ndarray) -> np.ndarray:
        """
        Map temperature field (in °C) to grayscale intensity based on sensor config.
        Linear mapping: min_temp -> 0, max_temp -> max_intensity.
        """
        cfg = self.config
        clipped = np.clip(temp_field, cfg.min_temp, cfg.max_temp)
        normalized = (clipped - cfg.min_temp) / (cfg.max_temp - cfg.min_temp + 1e-8)
        intensities = normalized * cfg.max_intensity
        return intensities.astype(np.float32)


class IRDatasetGenerator:
    """
    Orchestrates frame generation, sensor effects, and saving.
    """
    def __init__(
        self,
        config: IRSensorConfig,
        scene: IRScene,
        output_dir: str,
        num_frames: int = 30
    ):
        self.config = config
        self.scene = scene
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def apply_sensor_effects(self, intensity_frame: np.ndarray) -> np.ndarray:
        """
        Apply a series of sensor-like effects:
        - atmospheric attenuation
        - blur
        - noise
        - bit-depth quantization
        """
        cfg = self.config
        frame = intensity_frame.copy()

        # Atmospheric attenuation
        frame = frame * cfg.atmosphere_attenuation

        # Gaussian blur
        if cfg.blur_kernel_size > 1:
            ksize = (cfg.blur_kernel_size, cfg.blur_kernel_size)
            frame = cv2.GaussianBlur(frame, ksize, 0)

        # Add Gaussian noise
        noise = np.random.normal(0, cfg.noise_std, frame.shape).astype(np.float32)
        frame = frame + noise

        # Clip to valid range
        frame = np.clip(frame, 0, cfg.max_intensity)

        # Quantize to sensor bit depth (e.g., 8-bit)
        # (Already in 0..max_intensity, but we can simulate coarser quantization if needed)
        step = 1  # for example, 1 for 8-bit; use >1 to simulate fewer levels
        frame = (np.round(frame / step) * step).astype(np.uint8)

        return frame

    def generate_frame(self, frame_idx: int) -> np.ndarray:
        """
        Generate a single IR frame (uint8 grayscale).
        """
        background_temp = self.scene.generate_background_temperature()

        temp_field = background_temp
        total_frames = self.num_frames

        # render all objects
        for obj in self.scene.objects:
            temp_field = self.scene.render_object_on_temperature(
                temp_field,
                obj,
                frame_idx,
                total_frames
            )

        intensity_frame = self.scene.temperature_to_intensity(temp_field)
        final_frame = self.apply_sensor_effects(intensity_frame)
        return final_frame

    def save_frame(self, frame: np.ndarray, frame_idx: int):
        filename = f"frame_{frame_idx + 1:03d}.png"
        path = self.output_dir / filename
        cv2.imwrite(str(path), frame)

    def generate_dataset(self):
        print(f"Generating {self.num_frames} frames to {self.output_dir} ...")
        for idx in range(self.num_frames):
            frame = self.generate_frame(idx)
            self.save_frame(frame, idx)
        print("Done.")


def main():
    # 1) Configure sensor
    sensor_cfg = IRSensorConfig(
        width=256,
        height=256,
        min_temp=15.0,
        max_temp=60.0,
        bit_depth=8,
        noise_std=3.0,
        blur_kernel_size=5,
        atmosphere_attenuation=0.97,
    )

    # 2) Create objects (you can make 1–3 for variety)
    obj1 = IRObject(
        initial_center=(80, 130),
        size=(40, 60),
        base_temp=40.0,
        temp_variation=5.0,
        shape="rectangle",
    )

    obj2 = IRObject(
        initial_center=(180, 80),
        size=(30, 30),
        base_temp=50.0,
        temp_variation=8.0,
        shape="circle",
    )

    # 3) Create scene
    scene = IRScene(config=sensor_cfg, objects=[obj1, obj2])

    # 4) Dataset generator
    output_dir = os.path.join("output", "ir_frames")
    generator = IRDatasetGenerator(
        config=sensor_cfg,
        scene=scene,
        output_dir=output_dir,
        num_frames=30,  # between 15–30 as requested
    )

    # 5) Generate frames
    generator.generate_dataset()


if __name__ == "__main__":
    main()

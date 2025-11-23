import os
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

from PIL import Image, ImageFilter


@dataclass
class IRSensorConfig:
    width: int = 256
    height: int = 256

    # Temperature model (in °C)
    min_temp: float = 15.0      # minimum scene temperature
    max_temp: float = 60.0      # maximum scene temperature

    # Sensor characteristics
    bit_depth: int = 8          # grayscale 0..255
    noise_std: float = 3.0      # noise standard deviation in intensity domain
    blur_radius: float = 1.5    # Gaussian blur radius
    atmosphere_attenuation: float = 0.97  # intensity multiplier

    @property
    def max_intensity(self) -> int:
        return (2 ** self.bit_depth) - 1


class IRObject:
    """
    Represents a warm object in the IR scene.
    Supports rectangle and circle shapes.
    """
    def __init__(
        self,
        initial_center: Tuple[int, int],
        size: Tuple[int, int],
        base_temp: float,
        temp_variation: float,
        shape: str = "rectangle"
    ):
        self.initial_center = initial_center
        self.size = size
        self.base_temp = base_temp
        self.temp_variation = temp_variation
        self.shape = shape

    def get_center_at_frame(self, frame_idx: int, total_frames: int) -> Tuple[int, int]:
        """
        Simple motion: moves horizontally across the frame over time.
        """
        cx0, cy0 = self.initial_center
        if total_frames <= 1:
            return cx0, cy0

        progress = frame_idx / (total_frames - 1)  # 0..1
        # Move ±40 pixels horizontally around the original center
        dx = int(80 * (progress - 0.5))
        return cx0 + dx, cy0

    def get_temperature_at_frame(self, frame_idx: int, total_frames: int) -> float:
        """
        Sinusoidal temperature variation to simulate heating/cooling.
        """
        if total_frames <= 0:
            return self.base_temp

        phase = 2 * math.pi * frame_idx / total_frames
        delta = self.temp_variation * math.sin(phase)
        return self.base_temp + delta


class IRScene:
    """
    Creates the temperature field of the scene: background + objects.
    Uses pure Python lists instead of numpy.
    """
    def __init__(self, config: IRSensorConfig, objects: List[IRObject]):
        self.config = config
        self.objects = objects

    def generate_background_temperature(self) -> List[List[float]]:
        """
        Background: horizontal temperature gradient + small random noise.
        Returns a 2D list [row][col] of temperatures.
        """
        h, w = self.config.height, self.config.width
        min_temp = self.config.min_temp
        # background is slightly above min_temp
        bg_max_temp = self.config.min_temp + 5.0

        temp_field: List[List[float]] = []

        for y in range(h):
            row = []
            for x in range(w):
                # gradient from left (cooler) to right (warmer)
                t = x / (w - 1) if w > 1 else 0.0
                base_temp = min_temp + (bg_max_temp - min_temp) * t

                # small random noise
                noise = random.gauss(0.0, 0.3)
                row.append(base_temp + noise)
            temp_field.append(row)

        return temp_field

    def render_object_on_temperature(
        self,
        temp_field: List[List[float]],
        ir_object: IRObject,
        frame_idx: int,
        total_frames: int
    ) -> None:
        """
        In-place modification of temp_field to draw the object.
        """
        h, w = self.config.height, self.config.width
        cx, cy = ir_object.get_center_at_frame(frame_idx, total_frames)
        obj_temp = ir_object.get_temperature_at_frame(frame_idx, total_frames)

        obj_w, obj_h = ir_object.size

        y_min = max(cy - obj_h // 2, 0)
        y_max = min(cy + obj_h // 2, h)
        x_min = max(cx - obj_w // 2, 0)
        x_max = min(cx + obj_w // 2, w)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if ir_object.shape == "circle":
                    # inside circle?
                    if (x - cx) ** 2 + (y - cy) ** 2 <= (min(obj_w, obj_h) // 2) ** 2:
                        temp_field[y][x] = obj_temp
                else:
                    # rectangle or default
                    temp_field[y][x] = obj_temp

    def temperature_to_intensities(self, temp_field: List[List[float]]) -> List[int]:
        """
        Convert 2D temperature field to a flat list of grayscale intensities (0..255).
        Linear mapping between min_temp and max_temp.
        """
        cfg = self.config
        min_t = cfg.min_temp
        max_t = cfg.max_temp
        max_i = cfg.max_intensity

        pixels: List[int] = []
        for row in temp_field:
            for T in row:
                # clip to sensor range
                T_clipped = max(min_t, min(max_t, T))
                # normalize
                if max_t > min_t:
                    x = (T_clipped - min_t) / (max_t - min_t)
                else:
                    x = 0.0
                intensity = int(round(x * max_i))
                pixels.append(intensity)
        return pixels


class IRDatasetGenerator:
    """
    Orchestrates generation of frames and saving them as PNGs using Pillow.
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

    def apply_sensor_effects(self, pixels: List[int]) -> Image.Image:
        """
        Apply sensor effects in the intensity domain:
        - atmospheric attenuation
        - Gaussian noise
        - bit-depth quantization (via clipping & int cast)
        - Gaussian blur (via Pillow filter)
        """
        cfg = self.config
        max_i = cfg.max_intensity

        # 1. Atmospheric attenuation + noise + quantization
        processed: List[int] = []
        for val in pixels:
            # atmospheric attenuation
            v = val * cfg.atmosphere_attenuation

            # additive Gaussian noise
            v += random.gauss(0.0, cfg.noise_std)

            # clip and quantize
            v = max(0, min(max_i, int(round(v))))
            processed.append(v)

        # 2. Create image from processed intensities
        img = Image.new("L", (cfg.width, cfg.height))  # 'L' = 8-bit grayscale
        img.putdata(processed)

        # 3. Gaussian blur to mimic optics / diffusion
        if cfg.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))

        return img

    def generate_frame(self, frame_idx: int) -> Image.Image:
        """
        Generate a single IR frame as a Pillow Image.
        """
        # background
        temp_field = self.scene.generate_background_temperature()

        # overlay objects
        for obj in self.scene.objects:
            self.scene.render_object_on_temperature(
                temp_field, obj, frame_idx, self.num_frames
            )

        # convert temperature map to intensities
        pixels = self.scene.temperature_to_intensities(temp_field)

        # apply sensor effects
        img = self.apply_sensor_effects(pixels)
        return img

    def save_frame(self, img: Image.Image, frame_idx: int) -> None:
        filename = f"frame_{frame_idx + 1:03d}.png"
        filepath = self.output_dir / filename
        img.save(filepath)

    def generate_dataset(self) -> None:
        print(f"Generating {self.num_frames} frames in {self.output_dir} ...")
        for idx in range(self.num_frames):
            img = self.generate_frame(idx)
            self.save_frame(img, idx)
        print("Done.")


def main():
    # 1) Configure sensor
    cfg = IRSensorConfig(
        width=256,
        height=256,
        min_temp=15.0,
        max_temp=60.0,
        bit_depth=8,
        noise_std=3.0,
        blur_radius=1.5,
        atmosphere_attenuation=0.97,
    )

    # 2) Define objects
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

    # 3) Create scene with these objects
    scene = IRScene(config=cfg, objects=[obj1, obj2])

    # 4) Dataset generator
    output_dir = os.path.join("output", "ir_frames")
    generator = IRDatasetGenerator(
        config=cfg,
        scene=scene,
        output_dir=output_dir,
        num_frames=30,  # between 15 and 30
    )

    # 5) Generate frames
    generator.generate_dataset()


if __name__ == "__main__":
    main()

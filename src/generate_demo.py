from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo"
OUTPUT_PATH = DEMO_DIR / "conveyor_demo.mp4"


@dataclass
class Product:
    start_frame: int
    y: int
    width: int
    height: int
    speed: float
    color: tuple[int, int, int]

    def rect_at(self, frame_index: int) -> tuple[int, int, int, int]:
        x = int(-self.width - 80 + (frame_index - self.start_frame) * self.speed)
        y = self.y + int(math.sin((frame_index - self.start_frame) * 0.12) * 2)
        return x, y, x + self.width, y + self.height


def draw_conveyor(frame: np.ndarray, frame_index: int) -> None:
    height, width = frame.shape[:2]
    frame[:] = (34, 39, 44)

    cv2.rectangle(frame, (0, 120), (width, height - 120), (62, 66, 70), -1)
    cv2.rectangle(frame, (0, 124), (width, height - 124), (96, 100, 104), 2)

    for x in range(-90, width + 90, 90):
        offset = (frame_index * 9) % 90
        cv2.line(frame, (x + offset, 132), (x + offset - 80, height - 132), (48, 52, 56), 2)

    cv2.rectangle(frame, (0, 0), (width, 78), (22, 27, 33), -1)
    cv2.putText(
        frame,
        "Synthetic high-speed conveyor feed",
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (235, 239, 242),
        2,
        cv2.LINE_AA,
    )


def draw_product(frame: np.ndarray, product: Product, frame_index: int) -> None:
    x1, y1, x2, y2 = product.rect_at(frame_index)
    if x2 < 0 or x1 > frame.shape[1]:
        return

    shadow = np.zeros_like(frame)
    cv2.ellipse(
        shadow,
        ((x1 + x2) // 2, y2 + 14),
        (max(22, product.width // 2), 12),
        0,
        0,
        360,
        (0, 0, 0),
        -1,
    )
    frame[:] = cv2.addWeighted(frame, 1.0, shadow, 0.18, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), product.color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(frame, (x1 + 8, y1 + 10), (x2 - 10, y1 + 10), (255, 255, 255), 1, cv2.LINE_AA)


def make_products(total_frames: int, width: int) -> list[Product]:
    random.seed(7)
    colors = [
        (67, 160, 255),
        (91, 214, 139),
        (239, 113, 116),
        (236, 190, 72),
        (176, 130, 255),
    ]
    products: list[Product] = []
    frame = 12

    while frame < total_frames - 20:
        w = random.randint(52, 86)
        h = random.randint(42, 66)
        y = random.randint(185, 332)
        speed = random.uniform(12.0, 18.5)
        products.append(Product(frame, y, w, h, speed, random.choice(colors)))
        frame += random.randint(14, 30)

    # Add two close pairs to exercise tracking at speed.
    products.append(Product(80, 238, 72, 58, 17.0, colors[0]))
    products.append(Product(88, 304, 66, 52, 16.5, colors[2]))
    products.append(Product(155, 214, 78, 60, 18.2, colors[3]))
    products.append(Product(164, 318, 60, 48, 17.4, colors[1]))

    return [
        product
        for product in products
        if product.start_frame * product.speed < width + total_frames * product.speed
    ]


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    fps = 30
    seconds = 9
    total_frames = fps * seconds
    products = make_products(total_frames, width)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {OUTPUT_PATH}")

    for frame_index in range(total_frames):
        frame = np.empty((height, width, 3), dtype=np.uint8)
        draw_conveyor(frame, frame_index)
        for product in products:
            draw_product(frame, product, frame_index)
        writer.write(frame)

    writer.release()
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

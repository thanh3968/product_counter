from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Track:
    track_id: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    previous_x: int
    missed: int = 0
    counted: bool = False
    trail: list[tuple[int, int]] = field(default_factory=list)

    def update(self, centroid: tuple[int, int], bbox: tuple[int, int, int, int]) -> None:
        self.previous_x = self.centroid[0]
        self.centroid = centroid
        self.bbox = bbox
        self.missed = 0
        self.trail.append(centroid)
        self.trail = self.trail[-16:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count products crossing a conveyor line.")
    parser.add_argument("--video", default="demo/conveyor_demo.mp4", help="Input video path.")
    parser.add_argument("--output", default="demo/conveyor_counted.mp4", help="Annotated output video path.")
    parser.add_argument("--line-x", type=int, default=640, help="Vertical count line position.")
    parser.add_argument("--min-area", type=int, default=900, help="Minimum contour area.")
    parser.add_argument("--max-distance", type=float, default=90.0, help="Max centroid distance for tracking.")
    parser.add_argument("--display", action="store_true", help="Show a realtime preview window.")
    return parser.parse_args()


def detect_products(frame: np.ndarray, min_area: int) -> list[tuple[tuple[int, int], tuple[int, int, int, int]]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Products in the demo are bright and saturated; this also works as a practical
    # baseline for colored packaging on a darker conveyor.
    mask = cv2.inRange(hsv, (0, 45, 95), (179, 255, 255))
    mask[:115, :] = 0
    mask[-105:, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 22 or h < 22:
            continue
        centroid = (x + w // 2, y + h // 2)
        detections.append((centroid, (x, y, w, h)))

    detections.sort(key=lambda item: item[0][0])
    return detections


def distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def update_tracks(
    tracks: dict[int, Track],
    detections: list[tuple[tuple[int, int], tuple[int, int, int, int]]],
    next_id: int,
    max_distance: float,
) -> tuple[dict[int, Track], int]:
    unmatched_tracks = set(tracks.keys())
    unmatched_detections = set(range(len(detections)))

    candidates: list[tuple[float, int, int]] = []
    for track_id, track in tracks.items():
        for index, (centroid, _) in enumerate(detections):
            candidates.append((distance(track.centroid, centroid), track_id, index))
    candidates.sort(key=lambda item: item[0])

    for dist, track_id, detection_index in candidates:
        if dist > max_distance:
            break
        if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
            continue
        centroid, bbox = detections[detection_index]
        tracks[track_id].update(centroid, bbox)
        unmatched_tracks.remove(track_id)
        unmatched_detections.remove(detection_index)

    for track_id in list(unmatched_tracks):
        tracks[track_id].missed += 1
        if tracks[track_id].missed > 8:
            del tracks[track_id]

    for detection_index in unmatched_detections:
        centroid, bbox = detections[detection_index]
        tracks[next_id] = Track(next_id, centroid, bbox, centroid[0], trail=[centroid])
        next_id += 1

    return tracks, next_id


def annotate(frame: np.ndarray, tracks: dict[int, Track], line_x: int, count: int) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    cv2.line(output, (line_x, 100), (line_x, height - 95), (0, 245, 255), 4, cv2.LINE_AA)
    cv2.putText(output, "COUNT LINE", (line_x + 14, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 245, 255), 2)

    for track in tracks.values():
        x, y, w, h = track.bbox
        color = (56, 220, 92) if not track.counted else (0, 175, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        cv2.circle(output, track.centroid, 5, color, -1, cv2.LINE_AA)
        cv2.putText(output, f"ID {track.track_id}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for start, end in zip(track.trail, track.trail[1:]):
            cv2.line(output, start, end, color, 2, cv2.LINE_AA)

    panel = np.zeros((78, width, 3), dtype=np.uint8)
    panel[:] = (18, 23, 29)
    output[:78, :] = cv2.addWeighted(output[:78, :], 0.25, panel, 0.75, 0)
    cv2.putText(output, f"Products counted: {count}", (28, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 248, 250), 2)
    cv2.putText(output, "OpenCV centroid tracking", (width - 380, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 210, 235), 2)
    return output


def run_counter(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    tracks: dict[int, Track] = {}
    next_id = 1
    count = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detections = detect_products(frame, args.min_area)
        tracks, next_id = update_tracks(tracks, detections, next_id, args.max_distance)

        for track in tracks.values():
            if not track.counted and track.previous_x < args.line_x <= track.centroid[0]:
                track.counted = True
                count += 1

        annotated = annotate(frame, tracks, args.line_x, count)
        writer.write(annotated)

        if args.display:
            cv2.imshow("Product counter", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    print(f"Counted {count} products")
    print(f"Saved annotated video to {output_path}")
    return count


def main() -> None:
    args = parse_args()
    run_counter(args)


if __name__ == "__main__":
    main()

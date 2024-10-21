import torch
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import hashlib

# Settings
SHOW_ANNOTATIONS = True
SHOW_TRACKS = True
CONFIDENCE_THRESHOLD = 0.5
INIT_FRAMES_THRESHOLD = 4  # Initialize tracker after this many consecutive frames
IOU_THRESHOLD = 0.3

video_path = "demos/tracking/example.mp4"
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, "..", "..")
sys.path.append(os.path.abspath(scripts_dir))

model = YOLO("yolov8n.pt")


class TrackerManager:
    def __init__(self):
        self.trackers = []
        self.next_object_id = 0
        self.max_frames_missing = 10
        self.potential_trackers = []
        self.max_potential_missing_frames = (
            2  # Allow potential trackers to miss up to 2 frames
        )

    def add_tracker(self, tracker_info):
        self.trackers.append(tracker_info)

    def remove_lost_trackers(self):
        self.trackers = [
            t for t in self.trackers if t["missing_frames"] <= self.max_frames_missing
        ]


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def get_color_from_class(class_name):
    hash_object = hashlib.md5(class_name.encode())
    hex_color = hash_object.hexdigest()[:6]
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


tracker_manager = TrackerManager()
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = "demos/tracking/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, int(fourcc), int(fps), (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (640, 640))
    input_image = input_image / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = torch.tensor(input_image).unsqueeze(0).float()

    with torch.no_grad():
        results = model(input_image)

    detections = results[0].boxes

    tracker_manager.remove_lost_trackers()

    for tracker_info in tracker_manager.trackers:
        tracker = tracker_info["tracker"]
        success, box = tracker.update(frame)
        if success:
            tracker_info["bbox"] = box
        else:
            tracker_info["missing_frames"] += 1

    trackers_matched = set()
    detections_matched = set()

    processed_detections = []
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        confidence = detection.conf[0].cpu().numpy()
        class_id = int(detection.cls[0].cpu().numpy())
        class_name = model.names[class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            h, w, _ = frame.shape
            x1 = int(x1 / 640 * w)
            y1 = int(y1 / 640 * h)
            x2 = int(x2 / 640 * w)
            y2 = int(y2 / 640 * h)

            detection_box = (x1, y1, x2 - x1, y2 - y1)
            processed_detections.append(
                {
                    "box": detection_box,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "index": i,
                }
            )

    for det in processed_detections:
        best_iou = 0
        best_tracker = None
        for tracker_info in tracker_manager.trackers:
            if tracker_info["class_id"] != det["class_id"]:
                continue
            tracker_box = tracker_info["bbox"]
            overlap = iou(det["box"], tracker_box)
            if overlap > best_iou:
                best_iou = overlap
                best_tracker = tracker_info

        if best_iou > IOU_THRESHOLD and best_tracker:
            best_tracker["missing_frames"] = 0
            trackers_matched.add(best_tracker["object_id"])
            detections_matched.add(det["index"])
            best_tracker["tracker"].init(frame, det["box"])
            best_tracker["bbox"] = det["box"]
            x, y, w_box, h_box = [int(v) for v in det["box"]]
            color = (0, 255, 0)  # Green for matched
            if SHOW_TRACKS:
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(
                    frame,
                    f"ID: {best_tracker['object_id']}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    for tracker_info in tracker_manager.trackers:
        if tracker_info["object_id"] not in trackers_matched:
            tracker_info["missing_frames"] += 1
            if tracker_info["missing_frames"] <= tracker_manager.max_frames_missing:
                box = tracker_info["bbox"]
                x, y, w_box, h_box = [int(v) for v in box]
                color = (0, 0, 255)  # Red for unmatched
                if SHOW_TRACKS:
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    cv2.putText(
                        frame,
                        f"ID: {tracker_info['object_id']} (Missing)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

    unmatched_detections = [
        det for det in processed_detections if det["index"] not in detections_matched
    ]

    for potential_tracker in tracker_manager.potential_trackers:
        potential_tracker["missing_frames"] += 1

    for det in unmatched_detections:
        matched_potential_tracker = None
        for potential_tracker in tracker_manager.potential_trackers:
            if potential_tracker["class_id"] != det["class_id"]:
                continue
            overlap = iou(det["box"], potential_tracker["bbox"])
            if overlap > IOU_THRESHOLD:
                potential_tracker["counter"] += 1
                potential_tracker["bbox"] = det["box"]
                potential_tracker["missing_frames"] = 0  # Reset missing frames
                matched_potential_tracker = potential_tracker
                break

        if matched_potential_tracker is None:
            # Create new potential tracker
            potential_tracker = {
                "bbox": det["box"],
                "class_id": det["class_id"],
                "counter": 1,
                "missing_frames": 0,
            }
            tracker_manager.potential_trackers.append(potential_tracker)
        else:
            # Check if potential tracker should be promoted
            if matched_potential_tracker["counter"] >= INIT_FRAMES_THRESHOLD:
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, matched_potential_tracker["bbox"])
                tracker_manager.next_object_id += 1
                tracker_info = {
                    "tracker": tracker,
                    "object_id": tracker_manager.next_object_id,
                    "class_id": matched_potential_tracker["class_id"],
                    "missing_frames": 0,
                    "bbox": matched_potential_tracker["bbox"],
                }
                tracker_manager.add_tracker(tracker_info)
                tracker_manager.potential_trackers.remove(matched_potential_tracker)

    # Remove potential trackers that have been missing for too long
    tracker_manager.potential_trackers = [
        pt
        for pt in tracker_manager.potential_trackers
        if pt["missing_frames"] <= tracker_manager.max_potential_missing_frames
    ]

    # Show annotations for unmatched detections
    for det in unmatched_detections:
        if SHOW_ANNOTATIONS:
            x, y, w_box, h_box = [int(v) for v in det["box"]]
            color = get_color_from_class(det["class_name"])
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(
                frame,
                f"{det['class_name']} {det['confidence']:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    out.write(frame)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

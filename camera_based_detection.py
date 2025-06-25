 # make sure clip module doesn’t raise import errors
from ultralytics import YOLO
import cv2
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Initialize DeepSORT tracker
tracker = DeepSort(max_age=25, n_init=3, embedder="mobilenet", half=True)
confirmed_POI_ids = set()
frame_skip_interval = 1  # Skip every 4 frames for speed

# Model paths
weapon_model_path = "C:\\Users\\raghu\\brecourt\\best.pt"
pose_model_path = "C:\\Users\\raghu\\brecourt\\yolo11n-pose.pt"

# Load models
weapon_model = YOLO(weapon_model_path).half()
pose_model = YOLO(pose_model_path).half()

# Setup webcam feed
# Example stream URL from DroidCam or similar
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)  # or 25
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_idx = 0
empty_frame_count = 0
MAX_EMPTY_FRAMES = 5
last_valid_visual = None

# IOA function
def compute_ioa(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / (boxB_area + 1e-6)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

while cap.isOpened():
    ret, frame = cap.read()
    frame_idx += 1

    if not ret or frame is None:
        print(f"Skipped empty or bad frame at #{frame_idx}")
        empty_frame_count += 1
        if empty_frame_count >= MAX_EMPTY_FRAMES:
            print("Reached max empty frames — exiting.")
            break
        continue

    empty_frame_count = 0
    print(f"\nProcessing Frame #{frame_idx}")

    # if frame_idx % frame_skip_interval == 0:
    weapon_results = weapon_model.predict(frame, conf=0.5, imgsz=416)[0]
    pose_results = pose_model.predict(frame, conf=0.65, imgsz=416)[0]
    img_visual = pose_results.plot()

    weapon_boxes = weapon_results.boxes.xyxy.cpu().numpy()
    person_boxes = pose_results.boxes.xyxy.cpu().numpy()
    keypoints = pose_results.keypoints.data.cpu().numpy()

    # Draw weapon detections
    for box, cls, conf in zip(weapon_boxes,
                                weapon_results.boxes.cls.cpu().numpy(),
                                weapon_results.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        label = f"{weapon_model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img_visual, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_visual, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Prepare detections for DeepSORT
    detections = []
    for (px1, py1, px2, py2) in person_boxes:
        detections.append(([px1, py1, px2 - px1, py2 - py1], 0.5, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        px1, py1, px2, py2 = map(int, ltrb)
        matched_pose_idx = None

        # POI detection logic
        # for idx, (bx1, by1, bx2, by2) in enumerate(person_boxes):
        #     ioa_area = compute_ioa([px1, py1, px2, py2], [bx1, by1, bx2, by2])
        #     if ioa_area > 0.8:
        #         matched_pose_idx = idx
        #         break

        max_iou = 0

        for idx, (bx1, by1, bx2, by2) in enumerate(person_boxes):
            iou = compute_iou([px1, py1, px2, py2], [bx1, by1, bx2, by2])
            if iou > 0.3 and iou > max_iou:  # threshold and best match
                max_iou = iou
                matched_pose_idx = idx


        is_poi_ioa, is_poi_wrist = False, False
        if matched_pose_idx is not None:
            person_box = [px1, py1, px2, py2]
            is_poi_ioa = any(compute_ioa(person_box, weapon_box) > 0.5 for weapon_box in weapon_boxes)

            kp = keypoints[matched_pose_idx]
            for wrist_idx in [9, 10]:  # left and right wrists
                x, y, v = kp[wrist_idx]
                if v > 0.5:
                    for wx1, wy1, wx2, wy2 in weapon_boxes:
                        if wx1 <= x <= wx2 and wy1 <= y <= wy2:
                            is_poi_wrist = True
                            break
                if is_poi_wrist:
                    break

        if is_poi_ioa and is_poi_wrist:
            confirmed_POI_ids.add(track_id)

        if track_id in confirmed_POI_ids:
            cv2.rectangle(img_visual, (px1, py1), (px2, py2), (0, 125, 255), 2)
            cv2.putText(img_visual, f"POI-{track_id}", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    last_valid_visual = img_visual

    # Show last valid frame
    if last_valid_visual is not None:
        cv2.imshow("Live POI Detection", last_valid_visual)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

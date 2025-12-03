#!/usr/bin/env python3
"""
golden_face.py

Detect face + eyes (OpenCV Haar cascades) and compute how close the face is
to the golden ratio (phi = 1.618) using a few facial ratios.

Usage:
    python golden_face.py --image path/to/image.jpg
    python golden_face.py              # runs webcam
"""

import cv2
import numpy as np
import argparse
import math

# Golden ratio
PHI = 1.618033988749895

# Helper: euclidean distance
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Compute normalized "error" relative to PHI
def relative_error_ratio(r):
    return abs(r - PHI) / PHI

# Main measurement function: returns (score, details, annotated_image)
def analyze_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar cascades included with OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")
    # Note: mouth cascade sometimes less reliable; we'll try to use it if found.

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    annotated = img.copy()
    details = {}

    if len(faces) == 0:
        details['error'] = "No face detected"
        return None, details, annotated

    # For simplicity use the largest detected face
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x, y, w, h) = faces[0]
    face_box = (x, y, w, h)
    cx, cy = x + w // 2, y + h // 2

    # draw face box
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(annotated, "Face", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    face_width = w
    face_height = h

    details['face_width'] = float(face_width)
    details['face_height'] = float(face_height)

    # Detect eyes inside face ROI
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = annotated[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 10))

    # compute eye centers (take up to two largest eyes)
    eye_centers = []
    if len(eyes) > 0:
        eyes = sorted(eyes, key=lambda r: r[2]*r[3], reverse=True)[:2]
        for (ex, ey, ew, eh) in eyes:
            center = (x + ex + ew//2, y + ey + eh//2)
            eye_centers.append(center)
            cv2.circle(annotated, center, 3, (255,0,0), -1)
            cv2.rectangle(annotated, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255,0,0), 1)

    # Try detect mouth (often below nose; restrict search region)
    mouth_center = None
    if not mouth_cascade.empty():
        # Search in bottom half of face ROI
        mh_y1 = int(h * 0.55)
        mh_y2 = int(h * 0.95)
        roi_mouth_gray = roi_gray[mh_y1:mh_y2, 0:w]
        mouths = mouth_cascade.detectMultiScale(roi_mouth_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 15))
        if len(mouths) > 0:
            mouths = sorted(mouths, key=lambda r: r[2]*r[3], reverse=True)
            (mx, my, mw, mh) = mouths[0]
            # convert to original img coords
            mx_global = x + mx + mw // 2
            my_global = y + mh_y1 + my + mh // 2
            mouth_center = (mx_global, my_global)
            cv2.circle(annotated, mouth_center, 3, (0,0,255), -1)
            cv2.rectangle(annotated, (x+mx, y+mh_y1+my), (x+mx+mw, y+mh_y1+my+mh), (0,0,255), 1)

    # For nose/chin: approximate nose tip as top of lower half median point; chin as bottom center of face
    chin = (x + w//2, y + h)  # bottom center
    forehead = (x + w//2, y)  # top center (approx forehead/hairline)
    nose_tip = (x + w//2, y + int(h * 0.45))  # approx nose position

    cv2.circle(annotated, chin, 3, (0,255,255), -1)
    cv2.circle(annotated, forehead, 3, (0,255,255), -1)
    cv2.circle(annotated, nose_tip, 3, (0,255,255), -1)

    # Compute ratios:
    ratios = {}
    # 1) face_height / face_width
    r1 = face_height / float(face_width) if face_width != 0 else 0.0
    ratios['height_over_width'] = r1

    # 2) interocular distance / face_width (if two eyes found)
    if len(eye_centers) >= 2:
        ioc = dist(eye_centers[0], eye_centers[1])
        ratios['interocular_over_width'] = ioc / float(face_width) if face_width != 0 else 0.0
        # draw line between eyes
        cv2.line(annotated, eye_centers[0], eye_centers[1], (255,0,0), 2)
    else:
        ratios['interocular_over_width'] = None

    # 3) nose_to_chin / nose_to_forehead (vertical ratio) -> similar to classical golden face measurement
    nose_chin = dist(nose_tip, chin)
    nose_forehead = dist(nose_tip, forehead)
    r3 = nose_chin / nose_forehead if nose_forehead != 0 else None
    ratios['nose_chin_over_nose_forehead'] = r3
    # draw vertical lines
    cv2.line(annotated, nose_tip, chin, (0,255,255), 2)
    cv2.line(annotated, nose_tip, forehead, (0,255,255), 2)

    # 4) mouth_width / face_width (if mouth found)
    if mouth_center is not None:
        # estimate mouth width using mouth rectangle center's neighborhood heuristically:
        # not perfect: use mouth detection rectangle width if available above; for now use 0.25*face_width as fallback
        # We attempted to get mw earlier; if mouth detection used, get mw from mouths[0]
        # For simplicity set mouth_width to mw if detection succeeded above
        # NOTE: mouth detection sets `mouths` variable in local scope when found
        try:
            mw_val = mouths[0][2]
            ratios['mouth_over_width'] = mw_val / float(face_width)
            # draw a horizontal line approx mouth width
            mx0 = x + mouths[0][0]
            my0 = y + mh_y1 + mouths[0][1]
            cv2.line(annotated, (mx0, my0 + mouths[0][3]//2), (mx0 + mouths[0][2], my0 + mouths[0][3]//2), (0,0,255), 2)
        except Exception:
            ratios['mouth_over_width'] = None
    else:
        ratios['mouth_over_width'] = None

    # Now compare these ratios to PHI in a normalized manner.
    # The golden ratio typically >1; some of our measured ratios will be <1 (like interocular/width)
    # To compare, convert certain ratios to a comparable form: where possible, use reciprocal when <1
    # Strategy: for each numeric ratio r:
    #   - If r is None: skip
    #   - If r >= 1: use r directly
    #   - If r < 1: use 1 / r to lift small ratios to >1, enabling a comparison to PHI scale
    # This is a heuristic.
    usable_errors = []
    per_ratio_info = {}
    for k, r in ratios.items():
        if r is None or r == 0:
            per_ratio_info[k] = {'value': r, 'error': None}
            continue
        compare_r = r if r >= 1.0 else (1.0 / r)
        err = relative_error_ratio(compare_r)
        per_ratio_info[k] = {'value': float(r), 'scaled_for_phi': float(compare_r), 'error': float(err)}
        usable_errors.append(err)

    if len(usable_errors) == 0:
        details['error'] = "No usable facial measurements found"
        return None, details, annotated

    avg_error = float(sum(usable_errors) / len(usable_errors))
    # Convert to a score where smaller is better; also compute "percent closeness" where 100% means exact match
    percent_closeness = max(0.0, 100.0 * (1.0 - avg_error))

    details['ratios_raw'] = ratios
    details['per_ratio'] = per_ratio_info
    details['avg_relative_error'] = avg_error
    details['percent_closeness'] = percent_closeness

    # Add text overlay on annotated image summarizing results
    text1 = f"Avg rel error: {avg_error:.3f}   Golden closeness: {percent_closeness:.1f}%"
    cv2.putText(annotated, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
    cv2.putText(annotated, f"height/width: {r1:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
    if ratios['interocular_over_width'] is not None:
        cv2.putText(annotated, f"interocular/w: {ratios['interocular_over_width']:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 1)
    if r3 is not None:
        cv2.putText(annotated, f"nose-chin/nose-fore: {r3:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
    if ratios['mouth_over_width'] is not None:
        cv2.putText(annotated, f"mouth/w: {ratios['mouth_over_width']:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)

    return percent_closeness, details, annotated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=False, help="Path to input image. If omitted, webcam will be used.")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print("Failed to read image:", args.image)
            return
        score, details, annotated = analyze_image(img)
        if score is None:
            print("Analysis failed:", details.get('error', 'unknown'))
        else:
            print(f"Golden closeness: {score:.2f}%")
            print("Details:", details)
        # Show annotated
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # webcam mode
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open webcam.")
            return
        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            score, details, annotated = analyze_image(frame)
            display = annotated
            if score is not None:
                cv2.putText(display, f"Golden closeness: {score:.1f}%", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Webcam - Golden Face", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

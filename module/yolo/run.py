from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolo26n-pose.pt')

roi_x, roi_y, roi_w, roi_h = 0, 0, 200, 200
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Chạy mô hình
    results = model(frame, verbose=False)
    print(results)
    # 2. Vẽ toàn bộ keypoints và khung xương lên frame
    # plot() trả về ảnh đã vẽ sẵn tất cả keypoints
    annotated_frame = results[0].plot()
    print(annotated_frame)
    # 3. Logic kiểm tra cổ tay trong ROI
    hand_in_roi = False
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        for person_kps in keypoints:
            # Kiểm tra cổ tay trái (9) và phải (10)
            for wrist_idx in [9, 10]:
                x, y = person_kps[wrist_idx]
                if x == 0 and y == 0: continue
                
                if (roi_x < x < roi_x + roi_w) and (roi_y < y < roi_y + roi_h):
                    hand_in_roi = True

    # 4. Vẽ vùng ROI lên frame đã có keypoints
    cv2.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # 5. Hiển thị trạng thái
    text = "Co ban tay trong ROI" if hand_in_roi else "Khong co tay trong ROI"
    color = (0, 0, 255) if hand_in_roi else (0, 255, 0)
    cv2.putText(annotated_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('YOLO Pose Detection', annotated_frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 300 # Đặt lại vị trí ROI cho dễ nhìn

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    
    # Khởi tạo frame đen hoặc copy frame gốc để vẽ từ đầu (thay vì dùng .plot())
    annotated_frame = frame.copy()
    
    both_hand = False
    left_hand = False
    right_hand = False

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        person_kps = results[0].keypoints.xy[0].cpu().numpy()
        
        # Danh sách điểm cần theo dõi: 7(L-Elbow), 8(R-Elbow), 9(L-Wrist), 10(R-Wrist)
        # Chúng ta kiểm tra nếu điểm nằm trong ROI thì đánh dấu True
        points_to_check = {
            # 7: "left", 
            # 8: "right", 
            9: "left",    # Cẳng tay trái
            10: "right"  # Cẳng tay phải
        }
        
        for idx, side in points_to_check.items():
            x, y = person_kps[idx]
            if x > 0 and y > 0:
                # Vẽ điểm lên màn hình
                cv2.circle(annotated_frame, (int(x), int(y)), 8, (255, 255, 0), -1)
                
                # Kiểm tra trong ROI
                if (roi_x < x < roi_x + roi_w) and (roi_y < y < roi_y + roi_h):
                    cv2.circle(annotated_frame, (int(x), int(y)), 12, (0, 0, 255), 2)
                    if side == "left": left_hand = True
                    if side == "right": right_hand = True

    # Logic kiểm tra cả hai tay
    both_hand = left_hand and right_hand

    # Vẽ ROI
    cv2.rectangle(annotated_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # Hiển thị trạng thái
    status_text = ""
    if both_hand: status_text = "Ca hai tay trong ROI"
    elif left_hand: status_text = "Tay trai trong ROI"
    elif right_hand: status_text = "Tay phai trong ROI"
    else: status_text = "Khong co tay trong ROI"
    
    cv2.putText(annotated_frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('YOLO Pose Detection', annotated_frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
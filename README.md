import cv2
from cvzone.HandTrackingModule import HandDetector

# สร้างวัตถุสำหรับการจับภาพจากกล้อง
cap = cv2.VideoCapture(0)  # 0 คือ ID ของกล้องเริ่มต้น

# สร้างตัวตรวจจับมือ
detector = HandDetector(detectionCon=0.8, maxHands=2)  # detectionCon = ความแม่นยำ, maxHands = จำนวนมือสูงสุด

while True:
    # อ่านภาพจากกล้อง
    success, img = cap.read()
    if not success:
        break
    
    # ตรวจจับมือในภาพ
    hands, img = detector.findHands(img)  # hands จะเก็บข้อมูลเกี่ยวกับมือที่ตรวจจับได้

    if hands:
        for hand in hands:
            # ดึงข้อมูลเกี่ยวกับมือ
            lmList = hand["lmList"]  # ตำแหน่งจุด Landmarks (21 จุดบนมือ)
            bbox = hand["bbox"]  # ขอบเขตของมือ (x, y, w, h)
            centerPoint = hand["center"]  # จุดศูนย์กลางของมือ (x, y)
            handType = hand["type"]  # ซ้ายหรือขวา ("Left" หรือ "Right")

            # แสดงข้อมูลบนภาพ
            cv2.putText(img, f'{handType}', (bbox[0], bbox[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # แสดงภาพ
    cv2.imshow("Hand Detection", img)

    # ออกจากโปรแกรมเมื่อกดปุ่ม 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการใช้งานกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()

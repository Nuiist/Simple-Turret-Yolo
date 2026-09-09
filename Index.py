import cv2
from ultralytics import YOLO
import traceback
import numpy as np

model = YOLO("yolo26n.pt")
results = model.predict(
        source=0,
        device="cpu",
        show=True,
        stream=True,
    )
frame_count = 0

#++++++++++++++++++++++++++++++++++++       CONFIG         +++++++++++++++++++++++++++++++++++++++
frame_h = 480 
frame_w = 640 
target="person" #//Цель, по умолчанию стоит любая персона
target_speed = 10 # Скорость обновления в кадрах (т.е. 30 кадров / 10 = раз в 3 секунды)

#Возращает (dx, dy) - смещение обьекта в пикселях от центра
def get_delta_from_center(box_xyxy, frame_width, frame_height):
    center_x = frame_width / 2
    center_y = frame_height / 2
    x1, y1, x2, y2 = box_xyxy
    obj_center_x = (x1 + x2) / 2
    obj_center_y = (y1 + y2) / 2
    
    dx = obj_center_x - center_x
    dy = obj_center_y - center_y
    return dx, dy

#Заглушка, в теории здесь вызов на
def set_target (X, y):
    print(50*"-")
    print(f"\n {label}: dx={dx:+.1f}px, dy={dy:+.1f}px \n")
    print(50*"-")



try:
    print ("Start")
    for result in results:
                frame_count += 1               
                if frame_count % target_speed == 0:  
                    print(f"Обработано кадров: {frame_count}")
                    for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                        dx, dy = get_delta_from_center(box, frame_w, frame_h)
                        cls = int(result.boxes.cls[i].item())
                        label = result.names[cls]
                        if label == target:                     
                            set_target(dx, dy)

finally:
    print("=" * 50)
    input ()
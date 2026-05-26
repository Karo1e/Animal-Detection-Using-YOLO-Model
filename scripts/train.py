import os
import torch
from ultralytics import YOLO

def execute_model_training():
    """Hàm quản lý vòng đời huấn luyện mạng YOLO nhận diện 80 loài động vật."""
    target_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Hệ thống huấn luyện sử dụng nền tảng tăng tốc: {target_device}")

    # Khởi tạo mô hình với trọng số pretrained (tự động lưu vào thư mục weights/)
    model = YOLO("weights/yolov8n.pt")

    # Thiết lập đường dẫn trỏ tới tệp cấu hình dành riêng cho môi trường Kaggle
    # (Nếu chạy cục bộ để test code, hãy đổi thành "datasets/animal_data/data.yaml")
    yaml_config_path = "datasets/kaggle_data.yaml"

    print("Khởi động chu kỳ huấn luyện sâu...")
    results = model.train(
        data=yaml_config_path,
        epochs=150,
        imgsz=640,
        batch=16,
        device=target_device,
        lr0=0.01,
        project="runs",
        name="train_all_animals",
        # Các siêu tham số Tăng cường Dữ liệu (Data Augmentation)
        scale=0.5,
        hsv_h=0.015,
        # Thuật toán Dừng Sớm (Early Stopping)
        patience=25
    )
    
    print("Trọng số tối ưu đã được nạp tại: runs/train_all_animals/weights/best.pt")

if __name__ == "__main__":
    execute_model_training()
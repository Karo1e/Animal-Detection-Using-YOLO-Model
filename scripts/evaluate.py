import os
from ultralytics import YOLO

def evaluate_model_performance(weights_path, data_config):
    """Hàm tính toán mAP, Precision, Recall trên tập test/val."""
    
    print(f"Đang nạp trọng số mô hình từ: {weights_path}...")
    model = YOLO(weights_path)
    
    print("Bắt đầu quá trình đánh giá (Validation)...")
    # Thực thi luồng đánh giá tự động của hệ thống
    metrics = model.val(
        data=data_config,
        split="test",     # Chỉ định đánh giá trên tập 'test'. Có thể đổi thành 'val'
        project="runs",
        name="evaluate_results",
        conf=0.25         # Ngưỡng tin cậy (Confidence threshold) tối thiểu để tính toán
    )
    
    # Trích xuất và xuất báo cáo các chỉ số đo lường cốt lõi
    print("\n--- BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (METRICS) ---")
    print(f"Độ chuẩn xác trung bình (Precision): {metrics.box.mp:.4f}")
    print(f"Độ bao phủ trung bình (Recall): {metrics.box.mr:.4f}")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print("------------------------------------------")
    
    print("Biểu đồ và Ma trận nhầm lẫn (Confusion Matrix) đã được xuất tại: runs/evaluate_results/")

if __name__ == "__main__":
    # Cấu hình đường dẫn kiểm thử cục bộ
    # Giả định trọng số best.pt tải từ Kaggle về đã được đặt trong thư mục weights/
    best_weights_url = "weights/best.pt"  
    dataset_yaml = "datasets/animal_data/data.yaml"
    
    if os.path.exists(best_weights_url):
        evaluate_model_performance(best_weights_url, dataset_yaml)
    else:
        print("Lỗi hệ thống: Không tìm thấy tệp best.pt.")
        print("Vui lòng tải trọng số huấn luyện từ Kaggle về và đặt vào thư mục 'weights/'.")
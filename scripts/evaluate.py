import os
from ultralytics import YOLO

def evaluate_model_performance(weights_path, data_config):
    """
    Đánh giá hiệu năng của mô hình YOLOv8 trên tập kiểm định/kiểm thử.
    Tính toán và in các chỉ số mAP, Precision, Recall và lưu các biểu đồ kết quả.
    
    Args:
        weights_path (str): Đường dẫn tới tệp trọng số .pt của mô hình.
        data_config (str): Đường dẫn tới tệp cấu hình dữ liệu .yaml.
    """
    print(f"Nạp trọng số mô hình từ: {weights_path}...")
    model = YOLO(weights_path)
    
    print("Bắt đầu quá trình đánh giá hiệu năng (Validation)...")
    
    # Thực hiện đánh giá trên tập kiểm thử (split='test')
    metrics = model.val(
        data=data_config,
        split="test",     
        project="runs",
        name="evaluate_results",
        conf=0.25         
    )
    
    # Hiển thị các chỉ số đánh giá cốt lõi
    print("\n================ BÁO CÁO KẾT QUẢ ĐÁNH GIÁ ================")
    print(f"Độ chính xác trung bình (Precision): {metrics.box.mp:.4f}")
    print(f"Độ bao phủ trung bình (Recall): {metrics.box.mr:.4f}")
    print(f"Chỉ số mAP@50:                     {metrics.box.map50:.4f}")
    print(f"Chỉ số mAP@50-95:                  {metrics.box.map:.4f}")
    print("==========================================================")
    
    print("Biểu đồ đánh giá và Ma trận nhầm lẫn đã được lưu tại: runs/evaluate_results/")

if __name__ == "__main__":
    # Thiết lập đường dẫn tệp trọng số mô hình và cấu hình tập dữ liệu
    weights_path = "weights/best.pt"  
    dataset_yaml_path = "datasets/animal_data/data.yaml"
    
    if os.path.exists(weights_path):
        evaluate_model_performance(weights_path, dataset_yaml_path)
    else:
        print(f"Lỗi: Không tìm thấy file trọng số tại đường dẫn: {weights_path}")
        print("Vui lòng đảm bảo tệp trọng số đã được đặt trong thư mục 'weights/'.")
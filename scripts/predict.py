import os
import glob
from ultralytics import YOLO

def run_animal_inference(image_path, model_weights_path):
    """
    Thực hiện dự đoán đối tượng sử dụng mô hình YOLOv8 trên một ảnh hoặc toàn bộ thư mục.
    
    Args:
        image_path (str): Đường dẫn tới tệp ảnh hoặc thư mục chứa ảnh.
        model_weights_path (str): Đường dẫn tới tệp trọng số .pt của mô hình.
    """
    print(f"Đang tải trọng số mô hình từ: {model_weights_path}...")
    model = YOLO(model_weights_path)
    
    # Xác định danh sách file ảnh cần xử lý
    if os.path.isdir(image_path):
        print(f"Đang quét tìm kiếm tệp ảnh đệ quy tại: {image_path}...")
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        target_sources = []
        for ext in extensions:
            target_sources.extend(glob.glob(os.path.join(image_path, "**", ext), recursive=True))
    else:
        target_sources = [image_path]

    if not target_sources:
        print(f"Lỗi: Không tìm thấy hình ảnh phù hợp tại đường dẫn: {image_path}")
        return

    print(f"Tìm thấy {len(target_sources)} hình ảnh. Bắt đầu quá trình suy luận tuần tự...")
    
    # Xử lý tuần tự từng ảnh để tối ưu hóa bộ nhớ RAM
    for idx, img_file in enumerate(target_sources, start=1):
        file_name = os.path.basename(img_file)
        
        # Chạy dự đoán cho ảnh hiện tại
        predictions = model.predict(
            source=img_file, 
            conf=0.20,
            save=True, 
            project="runs/detect/runs", 
            name="",               
            exist_ok=True,
            verbose=False
        )
        
        # Xử lý và in kết quả phát hiện ra màn hình
        for result_obj in predictions:
            boxes_array = result_obj.boxes
            
            if len(boxes_array) == 0:
                print(f"[{idx}/{len(target_sources)}] File: {file_name} -> Không phát hiện đối tượng.")
            else:
                print(f"[{idx}/{len(target_sources)}] File: {file_name} -> Đang xử lý:")
                for box in boxes_array:
                    class_id = int(box.cls.item())
                    confidence_score = float(box.conf.item())
                    animal_species = model.names[class_id]
                    print(f"   + Nhãn: {animal_species.upper()} | Độ tin cậy: {confidence_score * 100:.2f}%")

if __name__ == "__main__":
    # Thiết lập đường dẫn dữ liệu kiểm thử và file trọng số
    test_folder_path = "datasets/animal_data/images/test"
    weights_path = "weights/best.pt"
    
    if os.path.exists(weights_path):
        run_animal_inference(test_folder_path, weights_path)
    else:
        print(f"Lỗi: Không tìm thấy file trọng số tại đường dẫn: {weights_path}")
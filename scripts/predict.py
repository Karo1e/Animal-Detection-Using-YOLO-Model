import os
import cv2
from ultralytics import YOLO

def run_animal_inference(image_path, model_weights_path):
    """Thực thi dự báo mạng nơ-ron lên đối tượng hình ảnh mới."""
    
    print(f"Đang tải trọng số từ: {model_weights_path}...")
    model = YOLO(model_weights_path)
    
    print("Đang quét không gian pixel ảnh...")
    predictions = model.predict(
        source=image_path, 
        conf=0.40, 
        save=True, 
        project="runs", 
        name="predict_results"
    )
    
    for result_obj in predictions:
        boxes_array = result_obj.boxes
        
        for box in boxes_array:
            bounding_box = box.xyxy.cpu().numpy()
            class_id = int(box.cls.item())
            confidence_score = float(box.conf.item())
            
            # Ánh xạ tên loài động vật từ 80 lớp
            animal_species = model.names[class_id]
            
            print(f"Phát hiện: {animal_species.upper()}")
            print(f"Độ Tin cậy: {confidence_score * 100:.2f}%")

if __name__ == "__main__":
    # Test thử một ảnh cục bộ (nếu chạy trên máy cá nhân)
    test_image_url = "datasets/animal_data/images/test/test_dog.jpg"
    
    # Trỏ đến tệp kết xuất thu được từ Kaggle
    weights_url = "runs/train_all_animals/weights/best.pt"
    
    if os.path.exists(weights_url):
        run_animal_inference(test_image_url, weights_url)
    else:
        print("Cảnh báo: Không thể tìm thấy tệp best.pt.")
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os

# Cấu hình giao diện và tiêu đề ứng dụng Streamlit
st.set_page_config(page_title="Animal Detection System", layout="wide")
st.title("Hệ thống Phát hiện và Phân loại Động vật Đa loài")

@st.cache_resource
def load_model():
    """
    Tải mô hình YOLOv8 từ file trọng số best.pt và lưu vào bộ nhớ cache.
    Tránh việc nạp lại mô hình sau mỗi lần người dùng tương tác.
    """
    weights_path = "weights/best.pt"
    if not os.path.exists(weights_path):
        return None
    return YOLO(weights_path)

# Khởi tạo thực thể mô hình YOLO
model = load_model()

if model is None:
    st.error("Lỗi: Không tìm thấy file trọng số 'best.pt' trong thư mục 'weights/'. Vui lòng kiểm tra lại cấu trúc thư mục.")
else:
    st.success("Mô hình đã được tải thành công. Hệ thống sẵn sàng hoạt động.")

# Component upload ảnh
uploaded_file = st.file_uploader("Tải lên hình ảnh cần nhận diện (Hỗ trợ JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh đầu vào từ file upload dưới dạng PIL Image
    image = Image.open(uploaded_file)
    
    # Thiết lập bố cục hiển thị dạng 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hình ảnh gốc")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Kết quả nhận diện")
        
        # Kích hoạt quá trình dự đoán khi nhấn nút
        if st.button("Bắt đầu nhận diện"):
            with st.spinner("Đang thực hiện nhận diện đối tượng..."):
                # Thực hiện dự đoán trên ảnh PIL (hệ màu RGB) với ngưỡng tin cậy mặc định 0.20
                results = model.predict(source=image, conf=0.20)
                
                # Trích xuất ảnh kết quả đã được vẽ bounding box (hệ màu mặc định của OpenCV là BGR)
                res_plotted = results[0].plot()
                
                # Chuyển đổi hệ màu BGR sang RGB để hiển thị chính xác trên Streamlit UI
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_plotted_rgb, use_container_width=True)
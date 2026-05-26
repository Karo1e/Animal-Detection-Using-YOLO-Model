import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# Cấu hình thông số trang web UI
st.set_page_config(page_title="Nhận diện Động vật YOLO", layout="wide")
st.title("Hệ thống Phát hiện và Phân loại Động vật đa loài 🐾")

@st.cache_resource
def load_model():
    """Nạp trọng số mạng nơ-ron vào bộ nhớ đệm (Cache) để tránh khởi tạo lại nhiều lần."""
    weights_path = "weights/best.pt"
    if not os.path.exists(weights_path):
        return None
    return YOLO(weights_path)

# Khởi tạo mô hình
model = load_model()

if model is None:
    st.error("Lỗi hệ thống: Không tìm thấy tệp 'best.pt' trong thư mục 'weights/'. Vui lòng huấn luyện mô hình trước!")
else:
    st.success("Tải trọng số mô hình thành công. Hệ thống đã sẵn sàng!")

# Khởi tạo bộ nạp tệp tin hình ảnh
uploaded_file = st.file_uploader("Tải lên một bức ảnh để nhận diện...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Giải mã tệp tin thành ma trận ảnh
    image = Image.open(uploaded_file)
    
    # Chia giao diện thành 2 cột song song
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Kết quả Dự báo")
        # Nút kích hoạt luồng xử lý
        if st.button("Khởi chạy Bộ lọc Nhận diện"):
            with st.spinner("Mạng nơ-ron đang phân tích không gian pixel..."):
                # Chuyển đổi định dạng PIL sang Numpy Array cho OpenCV và YOLO
                img_array = np.array(image)
                
                # Thực thi dự báo
                results = model.predict(source=img_array, conf=0.40)
                
                # Trích xuất ảnh ma trận đầu ra đã được render các hộp Bounding Box
                res_plotted = results[0].plot()
                
                # Chuyển đổi không gian màu từ BGR (OpenCV mặc định) sang RGB để hiển thị web
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_plotted_rgb, use_container_width=True)
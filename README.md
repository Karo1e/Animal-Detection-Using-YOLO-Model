# Hệ thống Phát hiện và Phân loại Động vật Đa loài (YOLOv8)

Dự án nghiên cứu đồ án GR1 tại Trường Công nghệ Thông tin và Truyền thông - Đại học Bách khoa Hà Nội (HUST). Ứng dụng mô hình học sâu một giai đoạn (YOLOv8) để nhận diện và phân loại tự động 80 loài động vật.

---

## 1. Tính năng cốt lõi

*   **Mô hình nhận diện:** Sử dụng kiến trúc YOLOv8 Nano (`YOLOv8n`) tối ưu hóa suy luận thời gian thực trên CPU.
*   **Tiền xử lý tối ưu:** Triển khai cơ chế xử lý song song đa luồng và Symbolic Link để xây dựng dataset chuẩn YOLO trên đám mây (Kaggle) giúp tiết kiệm dung lượng ổ đĩa.
*   **Kiểm thử tối ưu bộ nhớ:** Cơ chế suy luận tuần tự (sequential inference) giúp quét đệ quy các thư mục ảnh lớn tránh lỗi tràn bộ nhớ RAM.
*   **Ứng dụng Web trực quan:** Giao diện người dùng Streamlit hỗ trợ tải ảnh và trả kết quả nhận diện (đã tối ưu hóa hệ màu RGB/BGR).

## 2. Cấu trúc thư mục dự án

```text
├── app/
│   └── app.py             # Ứng dụng Web giao diện Streamlit
├── datasets/              # Tệp cấu hình dataset (data.yaml)
├── scripts/
│   ├── predict.py         # Tập lệnh suy luận ngoại tuyến tuần tự
│   └── evaluate.py        # Tập lệnh đánh giá hiệu năng (Precision/Recall/mAP)
├── weights/
│   └── best.pt            # File trọng số mô hình đã huấn luyện (YOLOv8n)
├── ngoai_le/              # Tài liệu LaTeX, biểu đồ và số liệu thực tế từ Kaggle
├── requirements.txt       # Danh sách thư viện phụ thuộc của dự án
└── README.md              # Tài liệu hướng dẫn sử dụng dự án
```

## 3. Cài đặt môi trường

Khuyến nghị sử dụng Python từ `3.8` đến `3.11`. Tiến hành cài đặt môi trường và các thư viện liên quan:

```bash
git clone https://github.com/Karo1e/Animal-Detection-Using-YOLO-Model
cd Animal-Detection-Using-YOLO-Model

python -m venv .venv
```

Kích hoạt môi trường ảo:
*   Trên Windows: `.venv\Scripts\activate`
*   Trên macOS/Linux: `source .venv/bin/activate`

Sau đó tiến hành cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

## 4. Hướng dẫn chạy chương trình

### 4.1. Khởi chạy Ứng dụng Web (Streamlit UI)
Chạy ứng dụng web giao diện người dùng trên trình duyệt cục bộ:
```bash
streamlit run app/app.py
```
Ứng dụng hoạt động tại địa chỉ mặc định `http://localhost:8501`.

### 4.2. Khởi chạy tập lệnh Dự đoán tuần tự (Offline Predict)
Để quét và nhận diện hàng loạt hình ảnh kiểm thử trong thư mục `datasets/animal_data/images/test/` (kết quả lưu tại `runs/detect/runs/`):
```bash
python scripts/predict.py
```

### 4.3. Đánh giá mô hình (Evaluation)
Để kiểm thử hiệu năng tổng quát (mAP@0.5, mAP@0.5:0.95, Precision, Recall) và xuất các biểu đồ đánh giá:
```bash
python scripts/evaluate.py
```

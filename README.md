# Phát hiện các loài động vật nhờ mô hình YOLO
(Animal Detection Using YOLO Model)

**Học phần:** Nghiên cứu tốt nghiệp 1  
**Đơn vị:** Đại học Bách khoa Hà Nội (HUST)  

---

## 1. Giới thiệu dự án
Dự án ứng dụng mô hình học sâu YOLO để giải quyết bài toán thị giác máy tính sinh thái: phát hiện và phân loại tự động 80 loài động vật khác nhau. 

Hệ thống được thiết kế theo cấu trúc module hóa chuẩn công nghiệp, kết hợp khả năng tính toán đám mây (Kaggle) để xử lý dữ liệu lớn trong pha huấn luyện và khả năng chạy suy luận nhẹ nhàng trên thiết bị cục bộ.

## 2. Cấu trúc thư mục cốt lõi
Dự án được phân tách rõ ràng giữa mã nguồn thực thi, dữ liệu và các tạo tác mô hình:
- `datasets/`: Cấu trúc trỏ đường dẫn dữ liệu (tệp `.yaml`).
- `scripts/`: Chứa mã nguồn logic nghiệp vụ chính (huấn luyện, dự báo).
- `weights/`: Nơi lưu trữ trọng số học chuyển giao (Transfer Learning).
- `runs/`: Thư mục tự động sinh ra chứa kết quả huấn luyện và ảnh đầu ra.
- `requirements.txt`: Khai báo khóa phiên bản môi trường thực thi.

## 3. Cài đặt môi trường
Đảm bảo hệ thống của bạn đã cài đặt Python (khuyến nghị >= 3.8). Khởi tạo môi trường và cài đặt các thư viện phụ thuộc bằng lệnh:

```bash
git clone <URL_GITHUB_CỦA_BẠN>
cd animal_detection_yolo
pip install -r requirements.txt
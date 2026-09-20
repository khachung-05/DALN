# DALN
# HỆ THỐNG NHÚNG BIÊN EDGE AI PHÁT HIỆN NGƯỜI BẰNG CẢM BIẾN NHIỆT ĐỘ PHÂN GIẢI THẤP

> **Đồ án môn học / Đồ án liên ngành**  
> **Sinh viên thực hiện:** Phạm Khắc Hùng  
> **Giáo viên hướng dẫn:** TS. Nguyễn Lệ Thu  
> **Nền tảng triển khai:** Raspberry Pi 5 & Cảm biến nhiệt Melexis MLX90640  

---

## 1. Giới thiệu bài toán & Mục tiêu đề tài

### 1.1. Bối cảnh và Thách thức
Trong các hệ thống giám sát an ninh và theo dõi con người hiện nay, các giải pháp truyền thống bộc lộ những rào cản lớn:
- **Xâm phạm quyền riêng tư:** Camera quang học (RGB) thông thường ghi lại chi tiết diện mạo, danh tính và hoạt động cá nhân, không thể triển khai tại các không gian nhạy cảm như phòng ngủ, phòng bệnh nhân hoặc khu vực chăm sóc người cao tuổi.
- **Hạn chế điều kiện môi trường:** Camera thường mất tác dụng trong điều kiện ánh sáng yếu, khói bụi hoặc bóng tối hoàn toàn (0 Lux).
- **Hạn chế phần cứng vi điều khiển truyền thống:** Các nghiên cứu trước đây (như bài báo gốc của Đại học KU Leuven) triển khai trên vi điều khiển STM32F4 buộc phải cắt tỉa mô hình mạng nơ-ron đến 136 lần để vừa bộ nhớ RAM nhỏ hẹp, khiến độ chính xác ở góc nghiêng $45^\circ$ bị sụt giảm nghiêm trọng (F1-score chỉ đạt $79.90\%$).

### 1.2. Mục tiêu đề tài
Xây dựng một hệ thống **Nhúng biên Trí tuệ nhân tạo (Embedded Edge AI)** hoàn chỉnh hoạt động độc lập (Offline):
- Tiếp cận theo nguyên lý **Privacy by Design**: Sử dụng cảm biến nhiệt độ phân giải siêu thấp ($32 \times 24$ điểm nhiệt) để nhận diện người dựa trên bức xạ hồng ngoại xa, hoàn toàn không thu thập hình ảnh chi tiết khuôn mặt.
- Ứng dụng mô hình học sâu hiện đại **YOLOv8-Nano** dạng Anchor-free để phát hiện người chính xác trong bóng tối tuyệt đối và khắc phục triệt để hiện tượng suy giảm nhận diện ở góc nghiêng.
- Tối ưu hóa mô hình bằng **ONNX Runtime** và kiến trúc xử lý đa luồng trên máy tính nhúng **Raspberry Pi 5** để đạt tốc độ xử lý thời gian thực mà không gây quá tải phần cứng.

---

## 2. Kiến trúc phần cứng hệ thống

Hệ thống được đóng gói và vận hành độc lập trên nền tảng phần cứng:
- **Bộ xử lý trung tâm (Edge Node):** Bo mạch máy tính nhúng **Raspberry Pi 5** (4 nhân ARM Cortex-A76 @ 2.4 GHz, RAM 2GB LPDDR4X).
- **Cảm biến đầu vào:** Cảm biến ma trận nhiệt hồng ngoại xa **Melexis MLX90640** (ma trận $32 \times 24 = 768$ điểm nhiệt, góc quan sát FOV $110^\circ \times 75^\circ$, tần số quét 8 Hz).
- **Giao tiếp phần cứng:** Giao thức **$I^2C$** phần cứng tốc độ cao Fast-mode Plus (tần số xung nhịp 1 MHz).
- **Hệ thống hiển thị & Điều khiển:** Màn hình cảm ứng điện dung 7 inch độ phân giải phần cứng $1024 \times 600$, kết nối tín hiệu qua cổng Micro-HDMI sang HDMI và nhận tín hiệu cảm ứng qua cáp USB HID.
- **Mạch nguồn:** Khối hạ áp Buck DC-DC chuyển đổi nguồn đầu vào 12V xuống 5V (dòng tải ổn định 5A/9A) chống sụt áp cho toàn bộ hệ thống.

---

## 3. Pipeline xử lý dữ liệu và Thuật toán AI

### 3.1. Tiền xử lý dữ liệu nhiệt (Pre-processing Pipeline)
Dữ liệu nhiệt độ thô ($32 \times 24$ số thực) từ cảm biến được đưa qua chuỗi xử lý toán học:
1. **Chuẩn hóa Min-Max thích nghi:** Tự động tìm dải nhiệt độ cao nhất ($T_{\text{max}}$) và thấp nhất ($T_{\text{min}}$) trong từng khung hình, co dãn dải nhiệt về khoảng $[0, 1]$ giúp thuật toán miễn nhiễm với biến thiên nhiệt độ môi trường.
2. **Lọc nhiễu trung vị (Median Filter $3 \times 3$):** Triệt tiêu các gai nhiễu nhiệt ngẫu nhiên phát sinh từ cảm biến ma trận.
3. **Nội suy song tuyến tính (Bilinear Interpolation):** Phóng đại ma trận nhiệt 16 lần từ $32 \times 24$ lên kích thước $512 \times 384$ để làm mịn các đường biên nhiệt độ.
4. **Chuẩn hóa đầu vào:** Chuyển đổi ma trận thành dạng ảnh 3 kênh màu và co dãn về kích thước $640 \times 640$ chuẩn hóa làm đầu vào cho mô hình AI.

### 3.2. Mô hình nhận diện đối tượng YOLOv8-Nano
- **Cấu trúc mạng:** Sử dụng kiến trúc **YOLOv8-Nano** dạng Anchor-free (loại bỏ cơ chế hộp neo cố định, dự đoán trực tiếp tâm và kích thước hộp bao) giúp bắt chính xác các khối nhiệt người bị biến dạng hình học.
- **Tập dữ liệu thực nghiệm:** Thu thập thực tế 1.369 khung hình ảnh nhiệt trong 3 bối cảnh (văn phòng, phòng học, hành lang) ở 2 góc đặt cảm biến ($45^\circ$ và $90^\circ$ từ trần nhà), gán nhãn thủ công trên nền tảng Roboflow.
- **Huấn luyện:** Áp dụng kỹ thuật học chuyển giao (Transfer Learning) từ trọng số `yolov8n.pt`, tối ưu hóa hàm mất mát và phân lớp trong 30 epoch.

### 3.3. Kỹ thuật tối ưu hóa thực thi tại biên (Edge Optimization)
- Đóng băng mô hình và chuyển đổi sang định dạng **ONNX Runtime** (`best.onnx`) hỗ trợ tập lệnh tăng tốc tính toán song song ARM NEON trên CPU Raspberry Pi 5.
- Thiết kế mô hình xử lý **Đa luồng bất đồng bộ (Multi-threading)**: Tách riêng luồng giao tiếp đọc dữ liệu cảm biến qua $I^2C$ và luồng chạy suy luận AI thông qua hàng đợi an toàn `Queue`, tối ưu hóa thời gian chờ (I/O Bound).

---

## 4. Thiết kế Giao diện giám sát & Quản lý Media (GUI)

Phần mềm giám sát được phát triển bằng ngôn ngữ **Python** sử dụng thư viện **Tkinter** kết hợp **OpenCV** và **PIL**, hiển thị toàn màn hình $1024 \times 600$:
- **Vùng hiển thị trực quan (Bên trái, $944 \times 600$ px):** Trình chiếu luồng video nhiệt thời gian thực, tự động vẽ khung chữ nhật (Bounding Box) bao quanh đối tượng người, hiển thị nhãn nhận diện, độ tin cậy (`conf`) và số lượng người đếm được (`human: {num}`).
- **Cột thanh công cụ điều khiển (Bên phải, rộng 80 px):** Bố trí 3 nút bấm cảm ứng hình tròn vẽ bằng Canvas:
  - 📷 **Nút Chụp ảnh:** Chụp và lưu khung hình hiện tại kèm bounding box vào thư mục `media/`.
  - 🎥 **Nút Quay video:** Ghi luồng video nhiệt vào bộ nhớ dưới định dạng `.mp4`.
  - 📁 **Nút Thư viện Media:** Mở cửa sổ quản lý danh sách các tệp đa phương tiện đã ghi lại, tích hợp module xem ảnh tĩnh (`Image Viewer`) và phát lại video (`Video Player`).
- **Cơ chế cập nhật sự kiện:** Sử dụng phương thức đệ quy thời gian `root.after()` của Tkinter chu kỳ ~30ms, duy trì giao diện mượt mà không bị đơ giật khi người dùng tương tác cảm ứng.

---

## 5. Kết quả thực nghiệm đạt được

| Chỉ số đánh giá | Kết quả đạt được | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- |
| **mAP@0.5** | **94.8%** | Độ chính xác trung bình toàn diện của mô hình phát hiện người |
| **Precision** | **93.1%** | Tỷ lệ dự đoán chính xác đối tượng người (hạn chế báo động giả) |
| **Recall** | **92.4%** | Khả năng bao phủ không bỏ sót đối tượng trong khung hình |
| **F1-Score (Góc $45^\circ$)** | **92.15%** | Vượt xa mức $79.90\%$ của bài báo gốc KU Leuven trên STM32F4 |
| **Tốc độ xử lý (FPS)** | **9.36 FPS** | Độ trễ toàn bộ chu trình chỉ $106.8\text{ ms}$/frame (vượt tần số 8 Hz của cảm biến) |
| **Mức chiếm dụng CPU** | **28% - 35%** | Hệ thống vận hành mát mẻ, CPU ổn định ở mức $52^\circ\text{C}$ |
| **Bộ nhớ RAM tiêu thụ** | **142 MB** | Tối ưu hóa cực tốt, chỉ chiếm một phần rất nhỏ trên thanh RAM 2GB |

---

## 6. Cấu trúc thư mục mã nguồn

```text
├── models/
│   ├── best.pt              # Trọng số mô hình YOLOv8-Nano (PyTorch)
│   └── best.onnx            # Mô hình đã tối ưu hóa đồ thị tính toán ONNX
├── src/
│   ├── capture_i2c.py       # Module cấu hình và đọc mảng 768 điểm từ MLX90640 qua I2C
│   ├── preprocess.py        # Module tiền xử lý: Min-Max, Median Filter, Bilinear Interpolation
│   ├── inference.py         # Pipeline suy luận đa luồng ONNX Runtime trên CPU ARM
│   ├── gui_main.py          # Giao diện chính Tkinter hiển thị màn hình 7 inch ($1024 \times 600$)
│   └── media_manager.py     # Module quản lý danh sách xem ảnh tĩnh và phát lại video
├── media/                   # Thư mục hệ thống tệp lưu trữ ảnh (.png) và video (.mp4)
├── requirements.txt         # Danh sách thư viện phụ thuộc (OpenCV, Pillow, Ultralytics, ONNX)
└── README.md                # Tài liệu thuyết minh dự án

# Phát hiện bệnh võng mạc tiểu đường dựa trên phân đoạn tổn thương và AI có khả năng giải thích

## Tổng quan

Dự án đề xuất một hệ thống phát hiện bệnh võng mạc tiểu đường (Diabetic Retinopathy - DR) từ ảnh đáy mắt (Fundus Image) theo hướng có khả năng giải thích (Explainable AI).

Thay vì dự đoán trực tiếp mức độ bệnh từ ảnh đầu vào, hệ thống sẽ:

1. Phân đoạn các tổn thương trên võng mạc.
2. Trích xuất các đặc trưng lâm sàng từ các tổn thương.
3. Dự đoán mức độ bệnh võng mạc tiểu đường.
4. Giải thích kết quả dự đoán dựa trên các triệu chứng đã phát hiện.

Hệ thống được triển khai dưới dạng ứng dụng web sử dụng Streamlit nhằm hỗ trợ trực quan hóa kết quả phân đoạn, thống kê tổn thương và dự đoán mức độ bệnh.

---

# Động lực nghiên cứu

Các mô hình học sâu hiện nay thường đạt độ chính xác cao trong bài toán phân loại bệnh võng mạc tiểu đường nhưng hoạt động như một "hộp đen" (Black Box), khiến bác sĩ khó hiểu được nguyên nhân dẫn đến kết quả dự đoán.

Để tăng tính minh bạch và khả năng giải thích, đề tài sử dụng các tổn thương võng mạc làm các khái niệm trung gian (Concepts):

* Vi phình mạch (Microaneurysms - MA)
* Xuất huyết (Hemorrhages - HE)
* Dịch tiết cứng (Hard Exudates - EX)
* Dịch tiết mềm (Soft Exudates - SE)

Đây là những dấu hiệu quan trọng thường được bác sĩ nhãn khoa sử dụng để chẩn đoán bệnh võng mạc tiểu đường.

---

# Kiến trúc hệ thống

```text
Ảnh đáy mắt
      │
      ▼
Mô hình phân đoạn tổn thương
      │
      ├── MA Mask
      ├── HE Mask
      ├── EX Mask
      └── SE Mask
      │
      ▼
Trích xuất đặc trưng
      │
      ▼
Mô hình phân loại
      │
      ▼
Dự đoán mức độ DR
```

---

# Bộ dữ liệu

## IDRiD Dataset

Bộ dữ liệu IDRiD (Indian Diabetic Retinopathy Image Dataset) cung cấp:

### Ảnh gốc

* Ảnh võng mạc màu chất lượng cao.

### Nhãn phân đoạn tổn thương

* Microaneurysms (MA)
* Hemorrhages (HE)
* Hard Exudates (EX)
* Soft Exudates (SE)

### Nhãn mức độ bệnh

| Mức độ | Ý nghĩa       |
| ------ | ------------- |
| 0      | Không mắc DR  |
| 1      | DR nhẹ        |
| 2      | DR trung bình |
| 3      | DR nặng       |
| 4      | DR tăng sinh  |

---

# Phương pháp thực hiện

## 1. Tiền xử lý ảnh

Các bước tiền xử lý bao gồm:

* Chuẩn hóa kích thước ảnh.
* Chuẩn hóa giá trị pixel.
* Tăng cường độ tương phản bằng CLAHE.
* Data Augmentation.

---

## 2. Phân đoạn tổn thương

Mô hình phân đoạn được huấn luyện để phát hiện đồng thời 4 loại tổn thương.

| Tổn thương     | Ký hiệu |
| -------------- | ------- |
| Vi phình mạch  | MA      |
| Xuất huyết     | HE      |
| Dịch tiết cứng | EX      |
| Dịch tiết mềm  | SE      |

Đầu ra của mô hình:

```text
(H, W, 4)
```

Trong đó mỗi kênh tương ứng với một loại tổn thương.

Hàm mất mát:

```text
Dice Loss + Binary Cross Entropy Loss
```

Các chỉ số đánh giá:

* Dice Score
* IoU
* Precision
* Recall

---

## 3. Trích xuất đặc trưng

Các mặt nạ tổn thương sau khi dự đoán sẽ được chuyển thành các đặc trưng lâm sàng:

* Diện tích tổn thương.
* Số lượng tổn thương.
* Mật độ tổn thương.
* Phân bố tổn thương trên võng mạc.

Ví dụ vector đặc trưng:

```text
[
 MA_area,
 MA_count,
 HE_area,
 HE_count,
 EX_area,
 EX_count,
 SE_area,
 SE_count
]
```

---

## 4. Phân loại mức độ bệnh

Các đặc trưng tổn thương được sử dụng để dự đoán mức độ bệnh võng mạc tiểu đường.

Các lớp đầu ra:

| Nhãn | Mức độ           |
| ---- | ---------------- |
| 0    | No DR            |
| 1    | Mild DR          |
| 2    | Moderate DR      |
| 3    | Severe DR        |
| 4    | Proliferative DR |

Các mô hình có thể sử dụng:

* Logistic Regression
* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)

Các chỉ số đánh giá:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# Khả năng giải thích (Explainability)

Khác với các mô hình phân loại truyền thống, hệ thống cung cấp lời giải thích dựa trên các tổn thương được phát hiện.

Ví dụ:

```text
Kết quả dự đoán:
Severe DR

Các tổn thương phát hiện được:
- 32 vùng MA
- 14 vùng HE
- 7 vùng EX
- 2 vùng SE
```

Nhờ đó người dùng có thể hiểu nguyên nhân dẫn đến kết quả dự đoán của mô hình.

---

# Ứng dụng Streamlit

Giao diện web cung cấp các chức năng:

## Tải ảnh lên

Người dùng tải ảnh võng mạc cần phân tích.

## Hiển thị kết quả phân đoạn

Bao gồm:

* Ảnh gốc.
* MA Mask.
* HE Mask.
* EX Mask.
* SE Mask.

## Dự đoán mức độ bệnh

Hiển thị:

* Mức độ DR dự đoán.
* Độ tin cậy của mô hình.

## Bảng giải thích

Hiển thị:

* Thống kê tổn thương.
* Biểu đồ đóng góp của từng tổn thương.
* Nhận xét lâm sàng.

---

# Cấu trúc thư mục dự án

```text
project/
│
├── app.py
│
├── models/
│   ├── segmentation_model.pt
│   └── classifier_model.pt
│
├── datasets/
│
├── notebooks/
│
├── utils/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   └── classification.py
│
├── assets/
│
├── requirements.txt
│
└── README.md
```

---

# Cài đặt

Clone dự án:

```bash
git clone <repository-url>
cd project
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Chạy ứng dụng:

```bash
streamlit run app.py
```

---

# Kết quả mong đợi

* Phân đoạn chính xác các tổn thương võng mạc.
* Dự đoán đáng tin cậy mức độ bệnh võng mạc tiểu đường.
* Cung cấp lời giải thích dễ hiểu dựa trên các triệu chứng lâm sàng.
* Xây dựng ứng dụng web trực quan phục vụ nghiên cứu và trình diễn.

---

# Hướng phát triển

* Học đa nhiệm (Multi-task Learning) cho phân đoạn và phân loại.
* Concept Bottleneck Network.
* Explainable AI dựa trên Attention.
* Kết hợp thêm các bộ dữ liệu lớn.
* Triển khai trong môi trường hỗ trợ chẩn đoán thực tế.

---

# Tác giả

Đồ án chuyên ngành Trí tuệ Nhân tạo

Đề tài:

Phát hiện bệnh võng mạc tiểu đường dựa trên phân đoạn tổn thương võng mạc và trí tuệ nhân tạo có khả năng giải thích.

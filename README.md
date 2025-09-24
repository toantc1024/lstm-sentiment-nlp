# Ứng dụng Phân tích Cảm xúc

Một ứng dụng web đơn giản được xây dựng bằng Streamlit để phân tích cảm xúc văn bản sử dụng mạng neural LSTM.

## Tính năng

- Phân tích cảm xúc theo thời gian thực (tích cực/tiêu cực)
- Trực quan hóa bằng biểu đồ đồng hồ tương tác
- Văn bản mẫu để kiểm tra nhanh
- Hiển thị điểm tin cậy
- Giao diện hiện đại, sạch sẽ

## Yêu cầu

- Python 3.11.0
- Các thư viện được liệt kê trong `requirements.txt`

## Cài đặt

1. Cài đặt các gói cần thiết:

```bash
pip install -r requirements.txt
```

2. Chạy ứng dụng:

```bash
streamlit run app.py
```

## Cách sử dụng

1. Nhập văn bản của bạn vào khu vực nhập liệu
2. Nhấp "Analyze Sentiment" (Phân tích Cảm xúc)
3. Xem kết quả với điểm tin cậy và trực quan hóa
4. Thử các văn bản mẫu để kiểm tra nhanh

## Cấu trúc Dự án

- `app.py` - Tệp ứng dụng chính
- `model_utils.py` - Các hàm tải mô hình và dự đoán
- `ui_components.py` - Các thành phần hiển thị giao diện
- `lstm_text_classifier.h5` - Mô hình LSTM đã huấn luyện
- `tokenizer.pkl` - Tokenizer văn bản
- `requirements.txt` - Các thư viện Python cần thiết
- `nlp-sentiment-analysis.ipynb` - Jupyter notebook để phân tích dữ liệu khám phá và huấn luyện mô hình

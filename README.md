# Fine-tuning PhoBERT cho Phân tích Cảm xúc Tiếng Việt (3 nhãn)

Đây là project fine-tuning mô hình **`vinai/phobert-base`**, một mô hình Transformer SOTA cho Tiếng Việt, cho nhiệm vụ **Phân loại Cảm xúc 3 nhãn**: Tích cực (POS), Tiêu cực (NEG), và Trung tính (NEU).

Mô hình được huấn luyện và đánh giá trên bộ dữ liệu dựa trên VLSP (được cung cấp dưới dạng file CSV trong thư mục `data/`) và đạt được kết quả **F1-score (weighted) khoảng 71%** trên tập kiểm thử (test set).

## 📊 Kết quả Nổi bật

* **Mô hình nền:** `vinai/phobert-base` (Sử dụng `AutoModelForSequenceClassification` từ Hugging Face Transformers).
* **Tập dữ liệu:** Dữ liệu CSV cục bộ (`train.csv`, `val.csv`, `test.csv`) với 3 nhãn (`POS`, `NEG`, `NEU`).
* **Kết quả (trên tập Test):**
    * **Loss:** ~0.69
    * **F1-score (Weighted):** **~0.71**

Mô hình cho thấy khả năng phân loại tốt trên các ví dụ thực tế:

* `"Phim này hay thật sự, nội dung sâu sắc."` -> **POS (Score: ~0.93)**
* `"Shop làm ăn chán quá, giao hàng thì lâu, sản phẩm thì vỡ."` -> **NEG (Score: ~0.88)**
* `"Điện thoại này sẽ ra mắt vào tháng 12 tới."` -> **NEU (Score: ~0.51)**

## 🚀 Cài đặt và Môi trường

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/thien2603/PhoBERT-Sentiment-Analysis.git](https://github.com/thien2603/PhoBERT-Sentiment-Analysis.git)
    cd PhoBERT-Sentiment-Analysis
    ```

2.  **Tạo môi trường ảo** (khuyến khích để quản lý dependencies):
    ```bash
    python -m venv venv
    ```
    * **Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **Windows:**
        ```bash
        venv\Scripts\activate
        ```

3.  **Cài đặt thư viện:** Các thư viện cần thiết được liệt kê trong `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *(Đảm bảo bạn đã kích hoạt môi trường ảo trước khi chạy lệnh này).*

## ▶️ Cách thực thi

1.  **Dữ liệu:** Dữ liệu huấn luyện (`train.csv`), kiểm thử (`val.csv`), và đánh giá cuối cùng (`test.csv`) đã có sẵn trong thư mục `data/`.
2.  **Huấn luyện & Đánh giá:** Mở và chạy toàn bộ các ô code trong file Jupyter Notebook `train.ipynb` (hoặc tên file notebook của bạn).
    * Quá trình này sẽ tự động tải dữ liệu, tiền xử lý, huấn luyện model, và đánh giá trên tập test.
    * Model fine-tuned tốt nhất cùng với tokenizer sẽ được lưu vào thư mục `./results_sentiment_3label/`. (Thư mục này được `.gitignore` bỏ qua).
3.  **Thử nghiệm:** Phần cuối cùng của notebook (`SECTION 4`) sẽ tải model đã lưu và sử dụng `pipeline` để bạn thử nghiệm dự đoán trên các câu văn mới.

## 🛠️ Quy trình Thực hiện

Notebook `train.ipynb` thực hiện các bước sau:

1.  **Tải và Chuẩn bị Dữ liệu (`SECTION 1`):**
    * Sử dụng `load_dataset` của Hugging Face `datasets` để đọc 3 file `.csv` từ thư mục `data/`.
    * Ánh xạ (map) các nhãn dạng chuỗi (`POS`, `NEG`, `NEU`) sang dạng số nguyên (`2`, `0`, `1`) sử dụng dictionary `label2id`.
    * Tải `AutoTokenizer` tương ứng với `vinai/phobert-base`.
    * Định nghĩa hàm `preprocess_function` để token hóa văn bản (padding/truncation đến `max_length=128`).
    * Áp dụng tokenization cho toàn bộ dataset bằng phương thức `.map()`.

2.  **Tải Model và Cấu hình Huấn luyện (`SECTION 2`):**
    * Tải `AutoModelForSequenceClassification` từ `vinai/phobert-base`, cấu hình cho 3 nhãn và đính kèm `id2label`, `label2id`.
    * Thiết lập `TrainingArguments`, bao gồm thư mục output, số epochs, batch size, các tham số đánh giá/lưu trữ (`do_eval`, `save_steps`), và tắt logging lên `wandb` (`report_to="none"`).
    * Định nghĩa hàm `compute_metrics` sử dụng `evaluate.load("f1")` với `average="weighted"` để tính F1-score trong quá trình đánh giá.
    * Khởi tạo `Trainer` với model, arguments, datasets (train và validation), và hàm compute_metrics.

3.  **Huấn luyện và Đánh giá (`SECTION 3`):**
    * Gọi `trainer.train()` để bắt đầu quá trình fine-tuning.
    * Sau khi huấn luyện, lưu model và tokenizer vào thư mục output bằng `trainer.save_model()` và `tokenizer.save_pretrained()`.
    * Gọi `trainer.evaluate()` trên tập `test` để có kết quả đánh giá cuối cùng trên dữ liệu unseen.

4.  **Thử nghiệm (Inference) (`SECTION 4`):**
    * Tải lại model và tokenizer đã lưu từ thư mục output một cách tường minh.
    * Tạo một `pipeline("text-classification")` từ model và tokenizer đã tải.
    * Thử nghiệm pipeline với các câu ví dụ để kiểm tra khả năng dự đoán của model.

## 💻 Công nghệ sử dụng

* Python 3.x
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* NumPy

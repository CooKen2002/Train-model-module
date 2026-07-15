import torch


class Config:
    # --- Đường dẫn Dataset ---
    CLEAN_DATASET_ROOT = "./module/whisper_train/dts/clean_dataset" 
    CLEAN_PROMPTS_FILE = "./module/whisper_train/dts/prompt.txt"
    NOISE_DATASET_ROOT = "./module/whisper_train/dts/noise_dataset"

    # --- Đường dẫn Output ---
    MODEL_OUTPUT_DIR = "./module/whisper_train/models"
    ARTIFACTS_OUTPUT_DIR = "./module/whisper_train/artifacts"
    
    # --- Thông số Audio & Text ---
    SAMPLE_RATE = 16000
    AUDIO_MAX_LENGTH = 320000
    TEXT_MAX_LENGTH = 200

    # --- Thiết lập Huấn luyện (Training Settings) ---
    BATCH_SIZE = 12  # Số lượng mẫu nạp vào GPU/CPU trong một bước (step)
    TRAIN_RATE = 0.97  # Tỉ lệ chia tập dữ liệu: 90% Train, 10% Validation
    SEED = 3407  # Số random seed huyền thoại giúp kết quả lặp lại được ổn định

    # Tự động chọn GPU nếu máy có card đồ họa, ngược lại dùng CPU.
    # Note: Đã sửa từ "cuda" thành "gpu" để tương thích chuẩn với PyTorch Lightning Trainer.
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "small"  # tiny / base / small / medium / large-v1 / large-v2 / large-v3
    LANG = "vi"  # Ngôn ngữ mục tiêu

    # --- Cấu hình Bộ tối ưu (Optimizer & Scheduler) ---
    learning_rate = 0.0001 # Tốc độ học (Lr) ban đầu
    weight_decay = 0.01  # Kỹ thuật L2 regularization giảm overfitting
    adam_epsilon = 1e-8  # Số rất nhỏ tránh lỗi chia cho 0 trong thuật toán Adam
    warmup_steps = 2  # Số step tăng dần Lr từ 0 đến learning_rate để ổn định ban đầu
    num_worker = 4  # Số luồng CPU dùng để load dữ liệu song song
    num_train_epochs = 10000000000 # Chạy hết toàn bộ dataset 10 lần
    gradient_accumulation_steps = (
        1  # Tích lũy gradient (1 tức là cập nhật trọng số ngay sau mỗi batch)
    )

    # --- Cấu hình Trộn nhiễu (Noise Augmentation) ---
    # Biên độ nhiễu (dB). Vì hệ thống thực tế đã qua bộ lọc DeepFilterNet2,
    # nên ở đây chỉ giả lập nhiễu nhẹ (SNR cao từ 15 đến 30dB) mô phỏng phần nhiễu còn sót lại.
    SNR_DB_RANGE = (15.0, 30.0)
    P_CLEAN = 0.35  # Tỉ lệ 35% giữ nguyên file âm thanh sạch, 65% còn lại sẽ bị trộn nhiễu ngẫu nhiên

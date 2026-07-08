import os
import sys
import glob
import json
import argparse
import torch
import whisper
import ctranslate2.converters
from transformers import WhisperConfig, WhisperForConditionalGeneration
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# FIX: cho phép chạy `python train.py` từ bất kỳ đâu (vd: từ repo root) mà vẫn import
# được config.py/model.py/data_utils.py nằm cùng thư mục với file này.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import prepare_datasets
from model import WhisperModelModule
from config import Config

BEST_CKPT_INFO_FILE = "best_checkpoint.json"


def build_hf_config_from_openai_dims(dims: dict) -> WhisperConfig:
    return WhisperConfig(
        vocab_size=dims["n_vocab"],
        num_mel_bins=dims["n_mels"],
        encoder_layers=dims["n_audio_layer"],
        encoder_attention_heads=dims["n_audio_head"],
        decoder_layers=dims["n_text_layer"],
        decoder_attention_heads=dims["n_text_head"],
        d_model=dims["n_audio_state"],
        encoder_ffn_dim=dims["n_audio_state"] * 4,
        decoder_ffn_dim=dims["n_text_state"] * 4,
        max_source_positions=dims["n_audio_ctx"],
        max_target_positions=dims["n_text_ctx"],
    )


def convert_openai_state_dict_to_hf(
    openai_sd: dict, n_audio_layer: int, n_text_layer: int
) -> dict:
    """Map state_dict OpenAI Whisper -> state_dict HuggingFace WhisperForConditionalGeneration.
    Ánh xạ theo đúng tên layer trong source code gốc của openai/whisper (whisper/model.py).
    """
    hf_sd = {}

    # --- Encoder ---
    hf_sd["model.encoder.conv1.weight"] = openai_sd["encoder.conv1.weight"]
    hf_sd["model.encoder.conv1.bias"] = openai_sd["encoder.conv1.bias"]
    hf_sd["model.encoder.conv2.weight"] = openai_sd["encoder.conv2.weight"]
    hf_sd["model.encoder.conv2.bias"] = openai_sd["encoder.conv2.bias"]
    hf_sd["model.encoder.embed_positions.weight"] = openai_sd[
        "encoder.positional_embedding"
    ]

    for i in range(n_audio_layer):
        p = f"encoder.blocks.{i}."
        q = f"model.encoder.layers.{i}."
        hf_sd[q + "self_attn.q_proj.weight"] = openai_sd[p + "attn.query.weight"]
        hf_sd[q + "self_attn.q_proj.bias"] = openai_sd[p + "attn.query.bias"]
        hf_sd[q + "self_attn.k_proj.weight"] = openai_sd[p + "attn.key.weight"]
        hf_sd[q + "self_attn.v_proj.weight"] = openai_sd[p + "attn.value.weight"]
        hf_sd[q + "self_attn.v_proj.bias"] = openai_sd[p + "attn.value.bias"]
        hf_sd[q + "self_attn.out_proj.weight"] = openai_sd[p + "attn.out.weight"]
        hf_sd[q + "self_attn.out_proj.bias"] = openai_sd[p + "attn.out.bias"]
        hf_sd[q + "self_attn_layer_norm.weight"] = openai_sd[p + "attn_ln.weight"]
        hf_sd[q + "self_attn_layer_norm.bias"] = openai_sd[p + "attn_ln.bias"]
        hf_sd[q + "fc1.weight"] = openai_sd[p + "mlp.0.weight"]
        hf_sd[q + "fc1.bias"] = openai_sd[p + "mlp.0.bias"]
        hf_sd[q + "fc2.weight"] = openai_sd[p + "mlp.2.weight"]
        hf_sd[q + "fc2.bias"] = openai_sd[p + "mlp.2.bias"]
        hf_sd[q + "final_layer_norm.weight"] = openai_sd[p + "mlp_ln.weight"]
        hf_sd[q + "final_layer_norm.bias"] = openai_sd[p + "mlp_ln.bias"]

    hf_sd["model.encoder.layer_norm.weight"] = openai_sd["encoder.ln_post.weight"]
    hf_sd["model.encoder.layer_norm.bias"] = openai_sd["encoder.ln_post.bias"]

    # --- Decoder ---
    hf_sd["model.decoder.embed_tokens.weight"] = openai_sd[
        "decoder.token_embedding.weight"
    ]
    hf_sd["model.decoder.embed_positions.weight"] = openai_sd[
        "decoder.positional_embedding"
    ]

    for i in range(n_text_layer):
        p = f"decoder.blocks.{i}."
        q = f"model.decoder.layers.{i}."
        hf_sd[q + "self_attn.q_proj.weight"] = openai_sd[p + "attn.query.weight"]
        hf_sd[q + "self_attn.q_proj.bias"] = openai_sd[p + "attn.query.bias"]
        hf_sd[q + "self_attn.k_proj.weight"] = openai_sd[p + "attn.key.weight"]
        hf_sd[q + "self_attn.v_proj.weight"] = openai_sd[p + "attn.value.weight"]
        hf_sd[q + "self_attn.v_proj.bias"] = openai_sd[p + "attn.value.bias"]
        hf_sd[q + "self_attn.out_proj.weight"] = openai_sd[p + "attn.out.weight"]
        hf_sd[q + "self_attn.out_proj.bias"] = openai_sd[p + "attn.out.bias"]
        hf_sd[q + "self_attn_layer_norm.weight"] = openai_sd[p + "attn_ln.weight"]
        hf_sd[q + "self_attn_layer_norm.bias"] = openai_sd[p + "attn_ln.bias"]

        hf_sd[q + "encoder_attn.q_proj.weight"] = openai_sd[
            p + "cross_attn.query.weight"
        ]
        hf_sd[q + "encoder_attn.q_proj.bias"] = openai_sd[p + "cross_attn.query.bias"]
        hf_sd[q + "encoder_attn.k_proj.weight"] = openai_sd[p + "cross_attn.key.weight"]
        hf_sd[q + "encoder_attn.v_proj.weight"] = openai_sd[
            p + "cross_attn.value.weight"
        ]
        hf_sd[q + "encoder_attn.v_proj.bias"] = openai_sd[p + "cross_attn.value.bias"]
        hf_sd[q + "encoder_attn.out_proj.weight"] = openai_sd[
            p + "cross_attn.out.weight"
        ]
        hf_sd[q + "encoder_attn.out_proj.bias"] = openai_sd[p + "cross_attn.out.bias"]
        hf_sd[q + "encoder_attn_layer_norm.weight"] = openai_sd[
            p + "cross_attn_ln.weight"
        ]
        hf_sd[q + "encoder_attn_layer_norm.bias"] = openai_sd[p + "cross_attn_ln.bias"]

        hf_sd[q + "fc1.weight"] = openai_sd[p + "mlp.0.weight"]
        hf_sd[q + "fc1.bias"] = openai_sd[p + "mlp.0.bias"]
        hf_sd[q + "fc2.weight"] = openai_sd[p + "mlp.2.weight"]
        hf_sd[q + "fc2.bias"] = openai_sd[p + "mlp.2.bias"]
        hf_sd[q + "final_layer_norm.weight"] = openai_sd[p + "mlp_ln.weight"]
        hf_sd[q + "final_layer_norm.bias"] = openai_sd[p + "mlp_ln.bias"]

    hf_sd["model.decoder.layer_norm.weight"] = openai_sd["decoder.ln.weight"]
    hf_sd["model.decoder.layer_norm.bias"] = openai_sd["decoder.ln.bias"]

    # lm head (proj_out) dùng chung weight với token embedding (tied weights ở cả 2 kiến trúc)
    hf_sd["proj_out.weight"] = openai_sd["decoder.token_embedding.weight"]

    return hf_sd


def whisper_train(cfg):
    # Cố định tất cả các nhân tố ngẫu nhiên (seed) trên cả CPU, GPU để kết quả chạy luôn đồng nhất
    seed_everything(cfg.SEED, workers=True)
    # Tạo thư mục lưu log và checkpoint nếu chưa tồn tại
    artifacts_output_dir = cfg.ARTIFACTS_OUTPUT_DIR
    log_output_dir = f"{artifacts_output_dir}/logs"
    check_output_dir = f"{artifacts_output_dir}/checkpoints"
    Path(log_output_dir).mkdir(exist_ok=True, parents=True)
    Path(check_output_dir).mkdir(exist_ok=True, parents=True)
    # 1. Gọi hàm chuẩn bị và phân chia dữ liệu
    train_clean, eval_clean, train_noise, eval_noise = prepare_datasets(cfg)
    # 2. Setup công cụ giám sát đồ thị TensorBoard
    tflogger = TensorBoardLogger(
        save_dir=log_output_dir, name="whisper_vi_mix", version="00001"
    )
    # Cấu hình bộ tự động lưu checkpoint: Theo dõi biến "val/loss".
    # Nếu epoch nào có val/loss thấp nhất (mode="min"), nó sẽ lưu lại và chỉ giữ tối đa 3 bản tốt nhất (save_top_k=3).
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}",
        filename="checkpoint-{epoch:04d}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )
    # 3. Khởi tạo Module Mô hình (file model.py), truyền toàn bộ cặp data vào phục vụ cho dataloader bên trong
    model = WhisperModelModule(
        cfg=cfg,
        train_clean_pairs=train_clean,
        train_noise_paths=train_noise,
        eval_clean_pairs=eval_clean,
        eval_noise_paths=eval_noise,
    )
    # 4. Khởi tạo bộ điều khiển tối cao Trainer của PyTorch Lightning
    trainer = Trainer(
        precision=(
            "16-mixed" if cfg.DEVICE == "gpu" else 32
        ),  # FIX: PyTorch Lightning 2.x khuyến nghị chuỗi "16-mixed" thay vì số nguyên 16
        # (int đã bị deprecate, dễ khác hành vi tuỳ version PL đang cài).
        accelerator=cfg.DEVICE,  # Chọn thiết bị phần cứng chạy
        max_epochs=cfg.num_train_epochs,  # Giới hạn số Epoch train
        accumulate_grad_batches=cfg.gradient_accumulation_steps,  # Gom bao nhiêu batch thì update trọng số 1 lần
        logger=tflogger,  # Đưa cấu hình logger vào
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],  # Đưa bộ lưu checkpoint và bộ theo dõi tốc độ học vào vòng lặp
    )

    print("Bắt đầu quá trình huấn luyện...")
    trainer.fit(model)

    # FIX (bug #1): lưu lại đường dẫn checkpoint TỐT NHẤT (val/loss thấp nhất) mà
    # ModelCheckpoint đã chọn, thay vì để convert() phải đoán bằng cách sort tên file
    # theo epoch (epoch mới nhất trong top-3 KHÔNG đồng nghĩa là loss thấp nhất).
    best_path = checkpoint_callback.best_model_path
    if best_path:
        with open(f"{check_output_dir}/{BEST_CKPT_INFO_FILE}", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_model_path": best_path,
                    "best_model_score": float(checkpoint_callback.best_model_score)
                    if checkpoint_callback.best_model_score is not None
                    else None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("Checkpoint tốt nhất (val/loss thấp nhất):", best_path)
    else:
        print("[WARN] checkpoint_callback.best_model_path rỗng -- convert() sẽ fallback về checkpoint mới nhất.")

    print(
        "\nHoàn tất training. Checkpoint đã lưu tại:", f"{check_output_dir}"
    )

# MARK: CONVERT
def convert(cfg):
    artifacts_output_dir = cfg.ARTIFACTS_OUTPUT_DIR
    models_output_dir = cfg.MODEL_OUTPUT_DIR
    checkpoint_dir = f"{artifacts_output_dir}/checkpoints"

    # FIX (bug #1): ưu tiên đọc checkpoint TỐT NHẤT (val/loss thấp nhất) đã được
    # whisper_train() ghi lại vào best_checkpoint.json. Trước đây code lấy checkpoint
    # cuối cùng theo thứ tự sort tên file (epoch lớn nhất trong top-3 đã lưu), điều này
    # KHÔNG đảm bảo là checkpoint có val/loss thấp nhất -> có thể convert nhầm model tệ hơn.
    best_info_path = f"{checkpoint_dir}/{BEST_CKPT_INFO_FILE}"
    BEST_CHECKPOINT_PATH = None
    if os.path.exists(best_info_path):
        with open(best_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        candidate = info.get("best_model_path")
        if candidate and os.path.exists(candidate):
            BEST_CHECKPOINT_PATH = candidate
            print("Dùng checkpoint tốt nhất theo val/loss (từ best_checkpoint.json):", BEST_CHECKPOINT_PATH)

    if BEST_CHECKPOINT_PATH is None:
        available_ckpts = sorted(glob.glob(f"{checkpoint_dir}/*.ckpt", recursive=True))
        if not available_ckpts:
            print(f"Không tìm thấy checkpoint nào trong {checkpoint_dir}!")
            return
        BEST_CHECKPOINT_PATH = available_ckpts[-1]
        print(
            "[WARN] Không tìm thấy best_checkpoint.json -- fallback về checkpoint mới nhất "
            "theo tên file (có thể KHÔNG phải checkpoint có val/loss thấp nhất):",
            BEST_CHECKPOINT_PATH,
        )

    print("\nSẽ convert checkpoint:", BEST_CHECKPOINT_PATH)

    OPENAI_WHISPER_PT_PATH = f"{models_output_dir}/pt/whisper_vi{cfg.MODEL_NAME}_finetuned.pt"
    HF_OUTPUT_DIR = f"{models_output_dir}/hf/whisper_vi{cfg.MODEL_NAME}_hf"
    CT2_OUTPUT_DIR = f"{models_output_dir}/faster_whisper/sks_whisper_vi{cfg.MODEL_NAME}_demo_v2"
    
    # 1. Bóc state_dict thuần Whisper khỏi checkpoint Lightning
    ckpt = torch.load(BEST_CHECKPOINT_PATH, map_location="cpu")
    openai_state_dict = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    assert (
        len(openai_state_dict) > 0
    ), "Không bóc được state_dict nào — kiểm tra lại key trong checkpoint."
    # Tải cấu hình kích thước mạng (dimensions) của bản gốc (theo MODEL_NAME) để lưu kèm vào file .pt mới
    base_model = whisper.load_model(cfg.MODEL_NAME)
    dims = base_model.dims.__dict__
    del base_model
    # Lưu lại thành file định dạng .pt chuẩn của thư viện openai-whisper gốc
    torch.save(
        {"dims": dims, "model_state_dict": openai_state_dict}, OPENAI_WHISPER_PT_PATH
    )
    print("Đã lưu PT format (OpenAI Whisper) tại:", OPENAI_WHISPER_PT_PATH)

    # Khởi tạo một khung cấu hình trống theo chuẩn cấu trúc HuggingFace từ thông số kích thước (dims) bóc được ở bước trước
    hf_config = build_hf_config_from_openai_dims(dims)
    # Khởi tạo một mô hình HuggingFace rỗng hoàn toàn với cấu hình trên
    hf_model = WhisperForConditionalGeneration(hf_config)
    # Chạy hàm ánh xạ tên các layer từ dict OpenAI sang tên layer của HuggingFace
    mapped_state_dict = convert_openai_state_dict_to_hf(
        openai_state_dict,
        n_audio_layer=dims["n_audio_layer"],
        n_text_layer=dims["n_text_layer"],
    )
    # Nạp các trọng số đã đổi tên vào mô hình HuggingFace rỗng
    # strict=False giúp hệ thống không bị crash nếu có một vài key cấu hình phụ lệch nhau, chỉ in ra cảnh báo [WARN]
    missing, unexpected = hf_model.load_state_dict(mapped_state_dict, strict=False)
    if missing or unexpected:
        print("[WARN] missing keys:", missing)
        print("[WARN] unexpected keys:", unexpected)
    # Lưu mô hình đã hoàn thiện theo format cấu trúc thư mục của HuggingFace (gồm file config.json và model.safetensors hoặc pytorch_model.bin)
    hf_model.save_pretrained(HF_OUTPUT_DIR)
    print("Đã lưu model HuggingFace tại:", HF_OUTPUT_DIR)

    # Lưu tokenizer/feature_extractor multilingual gốc (không phụ thuộc fine-tune, dùng được luôn).
    # Nếu máy không có mạng tới HuggingFace Hub, bước này có thể lỗi -> bỏ qua, faster-whisper
    # vẫn tự dùng tokenizer multilingual chuẩn của Whisper khi load.
    try:
        from transformers import WhisperTokenizer, WhisperFeatureExtractor

        WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny", language="vietnamese", task="transcribe"
        ).save_pretrained(HF_OUTPUT_DIR)
        WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny").save_pretrained(
            HF_OUTPUT_DIR
        )
    except Exception as e:
        print(
            f"[WARN] Không tải được tokenizer/feature_extractor từ Hub ({e}) -- bỏ qua, không ảnh hưởng faster-whisper."
        )

    # Lựa chọn kiểu nén (Quantization): Nếu máy chạy CPU, ép kiểu số thực 32-bit về dạng số nguyên INT8 (mô hình nhẹ đi 4 lần, tính toán siêu nhanh trên CPU).
    # Nếu máy có GPU (CUDA), dùng kiểu FLOAT16 (giảm nửa dung lượng, tối ưu phần cứng Tensor Cores của NVIDIA).
    QUANTIZATION = "int8" if not torch.cuda.is_available() else "float16"
    # Khởi tạo bộ chuyển đổi chỉ định thư mục nguồn là mô hình HuggingFace
    converter = ctranslate2.converters.TransformersConverter(HF_OUTPUT_DIR)
    # Thực hiện lệnh Convert cuối cùng xuất ra thư mục đích
    # force=True cho phép ghi đè nếu thư mục đích đã tồn tại trước đó
    converter.convert(CT2_OUTPUT_DIR, quantization=QUANTIZATION, force=True)

    print(f"\nHoàn tất convert sang faster-whisper! Thư mục: {CT2_OUTPUT_DIR}")
    print("Quantization dùng:", QUANTIZATION)


if __name__ == "__main__":
    # FIX: trước đây chọn chạy train hay convert bằng cách comment/uncomment dòng code,
    # rất dễ quên bật lại train() khi thật sự cần train -> giờ dùng --mode tường minh.
    parser = argparse.ArgumentParser(description="Whisper fine-tuning pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "convert", "both"],
        default="both",
        help="train: chỉ chạy training | convert: chỉ convert checkpoint có sẵn | both: train rồi convert luôn",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.mode in ("train", "both"):
        whisper_train(cfg)
    if args.mode in ("convert", "both"):
        convert(cfg)

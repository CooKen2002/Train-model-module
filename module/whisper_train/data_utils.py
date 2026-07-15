import random
from pathlib import Path
import numpy as np
import torch
import torchaudio.transforms as at
import soundfile as sf
import whisper
from tqdm import tqdm
import torchaudio


# Đọc file wav
def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
    # print(wave_path)
    # Dùng soundfile thay vì torchaudio.load() để tránh lỗi thiếu DLL torchcodec trên Windows.
    data, sr = sf.read(wave_path, dtype="float32", always_2d=True)
    # data, sr = torchaudio.load(wave_path)
    waveform = torch.from_numpy(data.T)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


# Trộn file âm thanh sạch với file tiếng nhiễu theo tỷ lệ (đo bằng $SNR$ - Tỷ lệ Tín hiệu trên Nhiễu).
def mix_with_noise(
    clean: torch.Tensor, noise: torch.Tensor, snr_db: float, rng: random.Random
) -> torch.Tensor: 
    clean = clean.flatten()
    noise = noise.flatten()
    if len(noise) == 0:
        return clean
    if len(noise) < len(clean):
        noise = noise.repeat(len(clean) // len(noise) + 1)
    if len(noise) > len(clean):
        start = rng.randint(0, len(noise) - len(clean))
        noise = noise[start : start + len(clean)]

    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean()
    if noise_power <= 1e-10 or clean_power <= 1e-10:
        return clean

    scale = torch.sqrt((clean_power / (10 ** (snr_db / 10.0))) / noise_power)
    mixed = clean * random.uniform(0,1) + noise * scale
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


# Quét toàn bộ thư mục dataset để lập chỉ mục file âm thanh.
def build_audio_index(dataset_root):
    audio_index = {}
    for wav_path in Path(dataset_root).rglob("*.wav"):
        audio_index[wav_path.stem] = str(wav_path)
    return audio_index


# Đọc và phân tách nội dung file văn bản chứa nhãn.
def load_prompts(prompts_file):
    pairs = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            audio_id, text = line.split("|")
            pairs.append((audio_id, text.strip()))
    return pairs


# Khớp nối âm thanh với văn bản tương ứng và thực hiện bộ lọc dữ liệu thô ban đầu.
# def get_clean_audio_transcript_pairs(
#     prompts_file, dataset_root, text_max_len=200, audio_max_len=480000, sr=16000
# ):
#     audio_index = build_audio_index(dataset_root)
#     print(audio_index[0])
#     prompt_pairs = load_prompts(prompts_file)
#     print(prompt_pairs[0])
#     pairs = []
#     missing = 0
#     skipped_len = 0
#     for audio_id, text in tqdm(prompt_pairs, desc="Loading clean dataset"):
#         audio_path = audio_index.get(audio_id)
#         if not audio_path:
#             missing += 1
#             continue
#         if len(text) > text_max_len:
#             skipped_len += 1
#             continue
#         audio = load_wave(audio_path, sample_rate=sr)[0]
#         if len(audio) > audio_max_len:
#             skipped_len += 1
#             continue
#         pairs.append((audio_id, audio_path, text))

#     print(
#         f"[CLEAN] Tổng dòng prompts: {len(prompt_pairs)} | Thiếu wav: {missing} | Bỏ do quá dài: {skipped_len} | Dùng được: {len(pairs)}"
#     )
#     return pairs

def load_pair(prompts_file):
    all_data = []
    with open(prompts_file, 'r') as f:
        data = f.readlines()
    # print(data[:2])
    for i in data:
        splt = i.split("|")
        # print(splt)
        path_, text = splt[0], splt[1]
        all_data.append([path_, text])
        # break
    return all_data
    
def get_clean_audio_transcript_pairs(
    prompts_file, dataset_root, text_max_len=200, audio_max_len=480000, sr=16000
):
    prompt_pairs = load_pair(prompts_file)
    pairs = []
    missing = 0
    skipped_len = 0
    for audio_path, text in tqdm(prompt_pairs, desc="Loading clean dataset"): # MARK: [:10]
        naudio_path = "/home/skysoft/share_folder/TuanNQ/1001/DTS/dolly/extracted/" + audio_path
        # FIX: audio_id phải là string để split_clean_by_speaker tách được speaker
        # (audio_id.split("_")[0]). Trước đó bị hardcode = 0 (int) -> crash.
        # Format thực tế: "<speaker_id>/<uuid>.wav" (ví dụ "0/520aa164-....wav") -> UUID không
        # mang thông tin speaker, nên lấy phần thư mục đầu (speaker_id) ghép với "_" để
        # split("_")[0] ra đúng speaker, đồng thời vẫn giữ unique theo từng file.
        speaker_id = Path(audio_path).parts[0] if "/" in audio_path else "0"
        audio_id = f"{speaker_id}_{Path(audio_path).stem}"
        if not audio_path:
            missing += 1
            continue
        if len(text) > text_max_len:
            skipped_len += 1
            continue
        audio = load_wave(naudio_path, sample_rate=sr)[0]
        if len(audio) > audio_max_len:
            skipped_len += 1
            continue
        pairs.append((audio_id, naudio_path, text))

    print(
        f"[CLEAN] Tổng dòng prompts: {len(prompt_pairs)} | Thiếu wav: {missing} | Bỏ do quá dài: {skipped_len} | Dùng được: {len(pairs)}"
    )
    return pairs


def get_noise_audio_paths(noise_dataset_root):
    noise_paths = [str(p) for p in Path(noise_dataset_root).rglob("*.wav")]
    print(f"[NOISE] Tổng số file noise: {len(noise_paths)}")
    return noise_paths


"""Chia tập dữ liệu CLEAN thành Train và Validation (dùng để theo dõi/chọn checkpoint trong lúc train, không phải test set đánh giá cuối cùng)."""


def split_clean_by_speaker(pairs, train_rate=0.9, seed=3407):
    """FIX: đảm bảo LUÔN có >=1 speaker (hoặc >=1 sample) cho eval, ngay cả khi
    dataset chỉ có 1 speaker duy nhất (ví dụ spk01_prompts.txt) -> fallback chia theo từng dòng.
    """
    speakers = sorted({audio_id.split("_")[0] for audio_id, _, _ in pairs})
    rng = random.Random(seed)

    if len(speakers) < 2:
        print(
            f"[WARN] Chỉ có {len(speakers)} speaker -> không thể chia eval theo speaker. "
            f"Fallback: chia theo từng dòng (random theo audio_id)."
        )
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_rate))
        n_train = min(n_train, len(shuffled) - 1) if len(shuffled) > 1 else n_train
        train_pairs = [(p, t) for _, p, t in shuffled[:n_train]]
        eval_pairs = [(p, t) for _, p, t in shuffled[n_train:]]
        return train_pairs, eval_pairs

    rng.shuffle(speakers)
    n_train = max(1, int(len(speakers) * train_rate))
    n_train = min(n_train, len(speakers) - 1)  # đảm bảo eval có ít nhất 1 speaker

    train_speakers, eval_speakers = set(speakers[:n_train]), set(speakers[n_train:])
    train_pairs = [(p, t) for sid, p, t in pairs if sid.split("_")[0] in train_speakers]
    eval_pairs = [(p, t) for sid, p, t in pairs if sid.split("_")[0] in eval_speakers]

    if len(eval_pairs) == 0:
        print(
            "[WARN] eval_pairs vẫn rỗng sau khi chia theo speaker -> kiểm tra lại audio_id format."
        )

    return train_pairs, eval_pairs


""" Xáo trộn danh sách các file âm thanh nhiễu và chia chúng thành hai tập dành riêng cho Train và Eval
    NOTE: Chia riêng noise cho train/eval -> đảm bảo lúc validate, model không bị "đánh giá" bằng đúng
    noise file đã thấy lúc train (tránh đánh giá lạc quan giả, không phản ánh đúng khả năng tổng quát hóa)."""


def split_noise(noise_paths, train_rate=0.9, seed=3407):
    rng = random.Random(seed)
    paths = noise_paths.copy()
    rng.shuffle(paths)
    n_train = max(1, int(len(paths) * train_rate))
    return paths[:n_train], paths[n_train:]


# Chuẩn bị dataset
def prepare_datasets(cfg):
    print("Preparing dataset...")
    all_clean = get_clean_audio_transcript_pairs(
        cfg.CLEAN_PROMPTS_FILE,
        cfg.CLEAN_DATASET_ROOT,
        cfg.TEXT_MAX_LENGTH,
        cfg.AUDIO_MAX_LENGTH,
        cfg.SAMPLE_RATE,
    )
    train_clean, eval_clean = split_clean_by_speaker(
        all_clean, cfg.TRAIN_RATE, cfg.SEED
    )
    all_noise = get_noise_audio_paths(cfg.NOISE_DATASET_ROOT)
    train_noise, eval_noise = split_noise(all_noise, cfg.TRAIN_RATE, cfg.SEED)

    print("TRAIN CLEAN AUDIO NUM: ", len(train_clean))
    print("EVAL CLEAN AUDIO NUM : ", len(eval_clean))
    print("TRAIN NOISE FILE NUM : ", len(train_noise))
    print("EVAL NOISE FILE NUM  : ", len(eval_noise))

    # Kiểm tra an toàn -> tránh mất công train xong mới phát hiện val_dataloader rỗng
    assert (
        len(train_clean) > 0
    ), "train_clean RỖNG -> kiểm tra lại CLEAN_DATASET_ROOT/CLEAN_PROMPTS_FILE"
    assert (
        len(eval_clean) > 0
    ), "eval_clean RỖNG -> ModelCheckpoint sẽ không lưu được! Kiểm tra số speaker/TRAIN_RATE."
    if len(train_noise) == 0 or len(eval_noise) == 0:
        print("[WARN] train hoặc eval noise rỗng -> sample sẽ luôn dùng audio sạch.")

    return train_clean, eval_clean, train_noise, eval_noise


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clean_pairs,
        noise_paths,
        tokenizer,
        sample_rate,
        snr_db_range,
        p_clean,
        deterministic=False,
        seed=3407,
    ):
        self.clean_pairs = clean_pairs
        self.noise_paths = noise_paths
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.snr_db_range = snr_db_range
        self.p_clean = p_clean
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self):
        return len(self.clean_pairs)

    # Xử lý dữ liệu thời gian thực khi nạp mẫu thứ idx
    def __getitem__(self, idx):
        audio_path, text = self.clean_pairs[idx]
        rng = random.Random(self.seed + idx) if self.deterministic else random
        clean = load_wave(audio_path, sample_rate=self.sample_rate).flatten()

        if len(self.noise_paths) > 0 and rng.random() >= self.p_clean:
            noise = load_wave(
                rng.choice(self.noise_paths), sample_rate=self.sample_rate
            ).flatten()
            audio = mix_with_noise(
                clean,
                noise,
                rng.uniform(*self.snr_db_range),
                rng if self.deterministic else random.Random(),
            )
        else:
            audio = clean

        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
        text_tokens = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(text)
        return {
            "input_ids": mel,
            "labels": text_tokens[1:] + [self.tokenizer.eot],
            "dec_input_ids": text_tokens,
        }


class WhisperDataCollatorWithPadding:
    def __call__(self, features):
        # Gộp các ảnh phổ âm thanh thành một khối Tensor duy nhất (input_ids)
        input_ids = torch.concat([f["input_ids"][None, :] for f in features])
        labels, dec_input_ids = [f["labels"] for f in features], [
            f["dec_input_ids"] for f in features
        ]
        # Tìm ra độ dài văn bản lớn nhất (max_len) xuất hiện trong Batch đó
        max_len = max([len(l) for l in labels] + [len(d) for d in dec_input_ids])
        # Tiến hành kỹ thuật đệm
        # Ma trận nhãn labels, những vị trí thiếu sót sẽ được điền giá trị -100 (để hàm CrossEntropyLoss biết đường bỏ qua không tính điểm phạt).
        labels = [
            np.pad(l, (0, max_len - len(l)), "constant", constant_values=-100)
            for l in labels
        ]
        # Ma trận đầu vào dec_input_ids, các vị trí thiếu sẽ được điền mã số 50257
        dec_input_ids = [
            np.pad(d, (0, max_len - len(d)), "constant", constant_values=50257)
            for d in dec_input_ids
        ]
        # Ép toàn bộ các mảng dữ liệu đã được đệm thành các Tensor PyTorch chính thống để sẵn sàng đẩy thẳng lên GPU xử lý song song
        batch = {"labels": labels, "dec_input_ids": dec_input_ids}
        batch = {
            k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()
        }
        batch["input_ids"] = input_ids
        return batch

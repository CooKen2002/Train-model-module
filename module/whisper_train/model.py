import torch
from torch import nn
import whisper
# import evaluate

import torchaudio.functional as F
from torchmetrics.text import CharErrorRate

from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import Config
from data_utils import SpeechDataset, WhisperDataCollatorWithPadding


class WhisperModelModule(LightningModule):
    def __init__(
        self,
        cfg: Config,
        train_clean_pairs=[],
        train_noise_paths=[],
        eval_clean_pairs=[],
        eval_noise_paths=[],
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Cấu hình giải mã (Decoding Options)
        self.options = whisper.DecodingOptions(
            language=cfg.LANG, without_timestamps=True
        )

        # 2. Tải kiến trúc mô hình OpenAI Whisper gốc
        self.model = whisper.load_model(cfg.MODEL_NAME)

        # 3. Khởi tạo Tokenizer chuyên dụng cho ngôn ngữ mục tiêu (tiếng Việt)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language=cfg.LANG, task=self.options.task
        )

        # 4. ĐÓNG BĂNG ENCODER (Kỹ thuật mấu chốt)
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False

        # 5. Hàm tính lỗi (Loss Function)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # 6. Các độ đo đánh giá chất lượng
        # self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = CharErrorRate()

        # Lưu trữ các tập đường dẫn dữ liệu để dùng cho DataLoader phía dưới
        self.__train_clean_pairs = train_clean_pairs
        self.__train_noise_paths = train_noise_paths
        self.__eval_clean_pairs = eval_clean_pairs
        self.__eval_noise_paths = eval_noise_paths

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        # Decoder nhận âm thanh + chữ mồi trước đó để dự đoán từ tiếp theo
        out = self.model.decoder(
            batch["dec_input_ids"].long(), self.model.encoder(batch["input_ids"])
        )

        # Tính toán độ lỗi toán học (Loss)
        # out.view(-1, out.size(-1)): Phẳng hóa ma trận xác suất từ dự đoán
        # labels.view(-1): Phẳng hóa mảng từ đáp án thực tế
        loss = self.loss_fn(out.view(-1, out.size(-1)), batch["labels"].long().view(-1))

        # Ghi nhận nhật ký (Log) giá trị loss lên thanh tiến trình và TensorBoard
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        out = self.model.decoder(
            batch["dec_input_ids"].long(), self.model.encoder(batch["input_ids"])
        )
        loss = self.loss_fn(out.view(-1, out.size(-1)), batch["labels"].long().view(-1))

        out_ids = torch.argmax(out, dim=2)
        out_ids[out_ids == -100] = self.tokenizer.eot
        labels_for_decode = batch["labels"].long().clone()
        labels_for_decode[labels_for_decode == -100] = self.tokenizer.eot

        o_list = [self.tokenizer.decode(o) for o in out_ids]
        l_list = [self.tokenizer.decode(l) for l in labels_for_decode]

        # Tính toán tỷ lệ sai chữ (CER) và sai từ (WER)
        # cer = self.metrics_cer.compute(o_list,l_list)
        # wer = F.word_error_rate(o_list,l_list)

        # log với on_epoch=True (mặc định của validation_step) để val/loss xuất hiện trong
        # callback_metrics ở cuối epoch -> ModelCheckpoint(monitor="val/loss") tìm thấy được.
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("val/cer", cer, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("val/wer", wer, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def lr_scheduler_step(self, scheduler, metric=None):
        # FIX: pytorch_lightning bản mới yêu cầu override hàm này khi dùng scheduler không
        # chuẩn API (LambdaLR từ get_linear_schedule_with_warmup của HuggingFace), nếu không
        # sẽ bị MisconfigurationException ngay khi gọi trainer.fit().
        scheduler.step()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.t_total = (
                (len(self.__train_clean_pairs) // self.cfg.BATCH_SIZE)
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def train_dataloader(self):
        dataset = SpeechDataset(
            self.__train_clean_pairs,
            self.__train_noise_paths,
            self.tokenizer,
            self.cfg.SAMPLE_RATE,
            snr_db_range=self.cfg.SNR_DB_RANGE,
            p_clean=self.cfg.P_CLEAN,
            deterministic=False,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg.num_worker,
            collate_fn=WhisperDataCollatorWithPadding(),
        )

    def val_dataloader(self):
        dataset = SpeechDataset(
            self.__eval_clean_pairs,
            self.__eval_noise_paths,
            self.tokenizer,
            self.cfg.SAMPLE_RATE,
            snr_db_range=self.cfg.SNR_DB_RANGE,
            p_clean=self.cfg.P_CLEAN,
            deterministic=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.num_worker,
            collate_fn=WhisperDataCollatorWithPadding(),
        )

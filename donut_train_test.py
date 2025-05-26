import os
from datasets import load_dataset
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch


# مسیر داده‌ها
data_root = "./donut_dataset"
model_name = "naver-clova-ix/donut-base"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)

# ✅ اضافه کردن تنظیمات ضروری برای آموزش
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id




def load_data(split):
    metadata_path = os.path.join(data_root, split, "metadata.json")
    ds = load_dataset("json", data_files=metadata_path)["train"]
    return ds
def preprocess(example, split):
    image_path = os.path.join(data_root, f"{split}/images", example["image"])
    image = Image.open(image_path).convert("RGB")

    # پردازش تصویر
    pixel_values = processor(image, return_tensors="pt").pixel_values[0]

    # ساخت دقیق input_ids
    encoding = processor.tokenizer(
        example["text_sequence"],
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoding.input_ids[0]
    print("MAX input_id:", input_ids.max().item())
    print("Vocab size:", processor.tokenizer.vocab_size)

    # جایگزینی pad_token_id با -100 (برای نادیده گرفتن در loss)
    input_ids[input_ids == processor.tokenizer.pad_token_id] = -100

    example["pixel_values"] = pixel_values
    example["labels"] = input_ids
    return example







train_ds = load_data("train").map(lambda x: preprocess(x, "train"), batched=False)
val_ds = load_data("val").map(lambda x: preprocess(x, "val"), batched=False)

# custom collator for pixel values
def collate_fn(batch):
    pixel_values = torch.stack([torch.tensor(b["pixel_values"]) if not isinstance(b["pixel_values"], torch.Tensor) else b["pixel_values"] for b in batch])
    labels = torch.stack([torch.tensor(b["labels"]) if not isinstance(b["labels"], torch.Tensor) else b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


args = Seq2SeqTrainingArguments(
    output_dir="./donut_model_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="epoch",
    fp16=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor,  # still needed for saving tokenizer config
)

trainer.train()

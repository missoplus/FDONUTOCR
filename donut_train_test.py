import os
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# ✅ Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Clear CUDA cache to avoid memory overload
torch.cuda.empty_cache()

# Select device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Using device:", device)

# Load model and processor
model_name = "naver-clova-ix/donut-base"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)

# Required model config for training
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.to(device)

# Dataset path
data_root = "./donut_dataset"

# Load data
def load_data(split):
    metadata_path = os.path.join(data_root, split, "metadata.json")
    return load_dataset("json", data_files=metadata_path)["train"]

# Preprocess each sample
def preprocess(example, split):
    image_path = os.path.join(data_root, f"{split}/images", example["image"])
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values[0]
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

    input_ids[input_ids == processor.tokenizer.pad_token_id] = -100
    example["pixel_values"] = pixel_values
    example["labels"] = input_ids
    return example

# Convert dataset using preprocessing
train_ds = load_data("train").map(lambda x: preprocess(x, "train"), batched=False)
val_ds = load_data("val").map(lambda x: preprocess(x, "val"), batched=False)

# Custom collator to create batches
def collate_fn(batch):
    pixel_values = torch.stack([
        b["pixel_values"] if isinstance(b["pixel_values"], torch.Tensor) else torch.tensor(b["pixel_values"])
        for b in batch
    ])
    labels = torch.stack([
        b["labels"] if isinstance(b["labels"], torch.Tensor) else torch.tensor(b["labels"])
        for b in batch
    ])
    return {
        "pixel_values": pixel_values,  # ❗ Do NOT move to device here
        "labels": labels
    }

# Training configuration
args = Seq2SeqTrainingArguments(
    output_dir="./donut_model_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,  # ❗ Use this if batch size can't be increased
    predict_with_generate=True,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    no_cuda=not torch.cuda.is_available()
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor
)

# 🚀 Start training
trainer.train()

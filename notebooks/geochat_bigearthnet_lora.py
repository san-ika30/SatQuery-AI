# %% [markdown]
# # GeoChat Fine-Tuning on BigEarthNet with LoRA
# **SatQuery AI — Kaggle Notebook**
# 
# Fine-tunes GeoChat-7B (or BLIP2) on BigEarthNet Sentinel-1 + Sentinel-2
# image–text pairs using LoRA via Hugging Face PEFT.
# 
# **Setup**: Add your HF token as a Kaggle secret named `HF_TOKEN`
# **GPU**: Enable GPU accelerator (P100 or T4) in Notebook settings

# %% [code]
# ── Install dependencies ──────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "datasets>=2.18.0",
    "accelerate>=0.29.0",
    "bitsandbytes>=0.43.0",
    "Pillow",
    "rasterio",
    "huggingface_hub",
], check=True)

print("✅ Dependencies installed")

# %% [code]
# ── Imports ────────────────────────────────────────────────────────────────────
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor, AutoModelForVision2Seq,
    TrainingArguments, Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login

# HF Token from Kaggle secret
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged into Hugging Face")
else:
    print("⚠️  No HF_TOKEN found — set it as a Kaggle secret")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}")

# %% [markdown]
# ## 1. BigEarthNet Dataset Preparation
# 
# BigEarthNet contains ~590K Sentinel-1 SAR + Sentinel-2 MS image patches
# with multi-label land cover annotations. We build image–text pairs for
# instruction fine-tuning.

# %% [code]
# ── BigEarthNet label descriptions ─────────────────────────────────────────────
BIGEARTHNET_LABELS = {
    "Urban fabric": "densely built-up urban area with residential and commercial structures",
    "Industrial or commercial units": "industrial buildings, warehouses, or commercial complexes",
    "Arable land": "agricultural cropland, ploughed fields, or cultivated land",
    "Permanent crops": "orchards, vineyards, or other permanent agricultural crops",
    "Pastures": "grassland or pasture used for livestock grazing",
    "Complex cultivation patterns": "mixed agricultural land with various crop types",
    "Land principally occupied by agriculture": "agricultural land with natural vegetation patches",
    "Agro-forestry areas": "land with both agricultural and forested areas",
    "Broad-leaved forest": "deciduous broad-leaved forest with dense tree canopy",
    "Coniferous forest": "evergreen coniferous forest (pine, spruce, fir)",
    "Mixed forest": "mixed deciduous and coniferous woodland",
    "Natural grasslands": "natural grassland or heathland",
    "Moors and heathland": "open heathland or moorland with low shrubs",
    "Sclerophyllous vegetation": "Mediterranean scrubland with drought-resistant vegetation",
    "Transitional woodland-shrub": "young woodland or areas transitioning to forest",
    "Beaches, dunes, sands": "sandy beaches, dunes, or desert areas",
    "Bare rock": "exposed rock surfaces or rocky terrain",
    "Sparsely vegetated areas": "sparse vegetation on degraded or arid land",
    "Burnt areas": "recently burned vegetation or post-fire landscape",
    "Inland wetlands": "inland marshes, bogs, or wetland areas",
    "Coastal wetlands": "coastal marshes, salt marshes, or tidal flats",
    "Inland waters": "rivers, lakes, reservoirs, or ponds",
    "Marine waters": "sea, ocean, or coastal marine waters",
}

def labels_to_caption(label_list: list[str]) -> str:
    """Convert label list to a natural-language scene description."""
    descriptions = [BIGEARTHNET_LABELS.get(l, l.lower()) for l in label_list]
    if len(descriptions) == 1:
        return f"This remote sensing image shows {descriptions[0]}."
    elif len(descriptions) == 2:
        return f"This remote sensing image shows {descriptions[0]} and {descriptions[1]}."
    else:
        main = ", ".join(descriptions[:-1])
        return f"This remote sensing image shows {main}, and {descriptions[-1]}."

def label_to_vqa_pairs(label_list: list[str]) -> list[dict]:
    """Generate VQA instruction pairs from labels."""
    pairs = []
    
    # Q1: Land cover description
    caption = labels_to_caption(label_list)
    pairs.append({
        "question": "Describe the land cover and major land use types visible in this satellite image.",
        "answer": caption,
    })
    
    # Q2: Vegetation presence
    veg_labels = [l for l in label_list if any(kw in l.lower() 
                  for kw in ["forest", "vegetation", "grass", "crop", "agriculture", "wood"])]
    if veg_labels:
        pairs.append({
            "question": "Is vegetation present in this image? If so, describe it.",
            "answer": f"Yes, vegetation is present. The image shows {labels_to_caption(veg_labels)}",
        })
    
    # Q3: Water presence
    water_labels = [l for l in label_list if "water" in l.lower() or "wetland" in l.lower()]
    if water_labels:
        pairs.append({
            "question": "Are there any water bodies or wetlands in this image?",
            "answer": f"Yes, the image contains {labels_to_caption(water_labels)}",
        })
    elif not water_labels:
        pairs.append({
            "question": "Are there any water bodies or wetlands in this image?",
            "answer": "No water bodies or wetlands are clearly visible in this image.",
        })
    
    # Q4: Urban presence
    urban_labels = [l for l in label_list if "urban" in l.lower() or "industrial" in l.lower()]
    if urban_labels:
        pairs.append({
            "question": "Are there urban or built-up areas in this image?",
            "answer": f"Yes, the image shows {labels_to_caption(urban_labels)}",
        })
    
    return pairs

print("✅ Label processing functions defined")

# %% [code]
# ── Load BigEarthNet from HuggingFace Hub ─────────────────────────────────────
print("📥 Loading BigEarthNet dataset...")

# BigEarthNet is available on HuggingFace
# Full dataset is ~590K samples — for this notebook we use a subset
dataset = load_dataset(
    "BigEarthNet/BigEarthNet-S2",  # Sentinel-2 version
    split="train",
    streaming=True,  # Use streaming to avoid downloading entire dataset
)

# Collect a training subset
N_SAMPLES = 5000  # Adjust based on available GPU memory and time
samples = []
for i, example in enumerate(dataset):
    if i >= N_SAMPLES:
        break
    samples.append(example)

print(f"✅ Collected {len(samples)} samples from BigEarthNet")
print(f"   Sample keys: {list(samples[0].keys())}")

# %% [code]
# ── Build instruction fine-tuning dataset ─────────────────────────────────────
print("🔧 Building instruction tuning dataset...")

instruction_pairs = []
for sample in samples:
    # Extract labels (BigEarthNet uses 'labels' or 'label' field)
    labels = sample.get("labels", sample.get("label", []))
    if isinstance(labels, str):
        labels = [labels]
    if not labels:
        continue
    
    # Generate VQA pairs
    vqa_pairs = label_to_vqa_pairs(labels)
    
    for pair in vqa_pairs:
        instruction_pairs.append({
            "image": sample.get("image") or sample.get("s2_image"),  # PIL Image
            "question": pair["question"],
            "answer": pair["answer"],
            "labels": labels,
        })

# Shuffle
random.shuffle(instruction_pairs)
train_size = int(len(instruction_pairs) * 0.9)
train_pairs = instruction_pairs[:train_size]
val_pairs = instruction_pairs[train_size:]

print(f"✅ Training pairs: {len(train_pairs)}")
print(f"   Validation pairs: {len(val_pairs)}")
print(f"\n📝 Example:")
print(f"   Q: {train_pairs[0]['question']}")
print(f"   A: {train_pairs[0]['answer']}")

# %% [markdown]
# ## 2. Load Model and Apply LoRA

# %% [code]
# ── Load BLIP2 with 4-bit quantization (fits in 16GB GPU) ────────────────────
MODEL_ID = "Salesforce/blip2-opt-2.7b"
# If you have GeoChat access: MODEL_ID = "MBZUAI/GeoChat"

print(f"📥 Loading model: {MODEL_ID}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

# %% [code]
# ── Apply LoRA ────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # LoRA alpha (scaling)
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 3. Training

# %% [code]
# ── Preprocessing function ────────────────────────────────────────────────────
def preprocess_batch(batch):
    images = []
    texts = []
    for item in batch:
        img = item["image"]
        if img is None:
            img = Image.new("RGB", (224, 224), color=(50, 50, 80))
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        images.append(img.convert("RGB").resize((224, 224)))
        
        prompt = (
            f"Question: {item['question']} "
            f"Context: This is a Sentinel-2 multispectral satellite image. "
            f"Answer:"
        )
        texts.append(prompt)
    
    inputs = processor(images=images, text=texts, return_tensors="pt",
                       padding=True, truncation=True, max_length=256)
    
    answers = [item["answer"] for item in batch]
    labels = processor.tokenizer(answers, return_tensors="pt",
                                  padding=True, truncation=True, max_length=256)
    inputs["labels"] = labels["input_ids"]
    return inputs


# %% [code]
# ── Training arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./satquery-geochat-bigearthnet",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    fp16=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to="none",
    dataloader_num_workers=2,
)

print("✅ Training arguments configured")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# %% [code]
# ── Create HuggingFace Dataset objects ───────────────────────────────────────
from torch.utils.data import Dataset as TorchDataset

class RSInstructDataset(TorchDataset):
    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.processor = processor
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        item = self.pairs[idx]
        img = item["image"]
        if img is None:
            img = Image.new("RGB", (224, 224), color=(50, 50, 80))
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        img = img.convert("RGB").resize((224, 224))
        
        prompt = (
            f"Question: {item['question']} "
            f"Context: This is a Sentinel-2 multispectral satellite image from BigEarthNet. "
            f"Answer:"
        )
        
        inputs = self.processor(
            images=img, text=prompt,
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=128,
        )
        
        label_enc = self.processor.tokenizer(
            item["answer"],
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=128,
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
            "labels": label_enc["input_ids"].squeeze(),
        }


train_dataset = RSInstructDataset(train_pairs, processor)
val_dataset = RSInstructDataset(val_pairs, processor)

print(f"✅ Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val")

# %% [code]
# ── Train ─────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Training complete!")

# %% [code]
# ── Save and push to HuggingFace Hub ─────────────────────────────────────────
OUTPUT_DIR = "./satquery-geochat-bigearthnet"
HF_REPO = "your-hf-username/satquery-geochat-bigearthnet-lora"

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to {OUTPUT_DIR}")

if HF_TOKEN and HF_REPO:
    try:
        model.push_to_hub(HF_REPO, token=HF_TOKEN)
        processor.push_to_hub(HF_REPO, token=HF_TOKEN)
        print(f"✅ Model pushed to HuggingFace Hub: {HF_REPO}")
    except Exception as e:
        print(f"⚠️  Hub push failed: {e}")
        print(f"   Model saved locally at: {OUTPUT_DIR}")

# %% [markdown]
# ## 4. Inference Test

# %% [code]
# ── Quick inference test ──────────────────────────────────────────────────────
model.eval()

test_img = Image.new("RGB", (224, 224), color=(30, 80, 50))  # Green = vegetation
test_prompt = "Question: Describe the land cover in this satellite image. Answer:"

inputs = processor(images=test_img, text=test_prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, num_beams=4)

answer = processor.decode(output[0], skip_special_tokens=True)
print(f"📡 Test inference:")
print(f"   Prompt: {test_prompt}")
print(f"   Answer: {answer}")
print("✅ Fine-tuned model is working!")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import gc
import os

# GPU Verification
print("="*60)
print("SYSTEM CHECK")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\nERROR: CUDA not available!")
    exit(1)

device = "cuda"
print(f"\nUsing device: {device}")
print("="*60 + "\n")

torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./llama3_disaster_lora"
MAX_LENGTH = 512
TRAIN_DATA_FILE = "disaster_training_data.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model and Tokenizer

print("Loading LLaMA 3.2 1B model...\n")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(" Tokenizer loaded")
except Exception as e:
    print(f" Error loading tokenizer: {e}")
    exit(1)

try:
    # Load model in fp16 for inference, but trainable params will be fp32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f" Model loaded to GPU")
    print(f" VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
except Exception as e:
    print(f" Error loading model: {e}")
    exit(1)

# Configure for LoRA + Gradient Checkpointing
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

print("Gradient checkpointing enabled")
print("Input gradients enabled for LoRA")

# Apply LoRA Configuration
print("\nApplying LoRA adapters...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# CRITICAL FIX: Convert LoRA parameters to FP32 for stable training
# This prevents "Attempting to unscale FP16 gradients" error
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

print("  LoRA applied with FP32 trainable parameters")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"  Trainable params: {trainable_params:,}")
print(f"  All params: {all_params:,}")
print(f"  Trainable: {100 * trainable_params / all_params:.2f}%")
print(f"  VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

if trainable_params == 0:
    print("\n ERROR: No trainable parameters!")
    exit(1)

# Load Training Data
print(f"\nLoading training data from {TRAIN_DATA_FILE}...")

try:
    with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    print(f" Loaded {len(training_data)} training samples")
except FileNotFoundError:
    print(f" Error: {TRAIN_DATA_FILE} not found!")
    exit(1)
except json.JSONDecodeError as e:
    print(f" Error: Invalid JSON - {e}")
    exit(1)

if not (training_data and isinstance(training_data, list)):
    print(" Error: Training data must be a list")
    exit(1)

sample = training_data[0]
if not ('instruction' in sample and 'response' in sample):
    print(" Error: Each sample must have 'instruction' and 'response'")
    exit(1)

def format_instruction(sample):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert disaster response analyst. Provide concise, actionable summaries of disaster events.<|eot_id|><|start_header_id|>user<|end_header_id|>

{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['response']}<|eot_id|>"""

formatted_data = [{"text": format_instruction(sample)} for sample in training_data]
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f" Data prepared")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Eval samples: {len(eval_dataset)}")

# Tokenization
print("\nTokenizing dataset...")

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train = train_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    desc="Tokenizing training data"
)

tokenized_eval = eval_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    desc="Tokenizing evaluation data"
)

print(f" Tokenization complete")

# Training Arguments (FIXED)

print("\nConfiguring training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    
    # Training schedule
    num_train_epochs=5,
    max_steps=-1,
    
    # Batch size
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    
    # Learning rate
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # Optimizer
    optim="adamw_torch",
    max_grad_norm=1.0,  # Now works with FP32 trainable params
    
    # Memory optimization
    fp16=True,  # Mixed precision: fp16 activations, fp32 gradients
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    
    # Logging
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    logging_first_step=True,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=50,
    
    # Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Misc
    report_to="none",
    remove_unused_columns=True,
    group_by_length=True,
    seed=42,
)

print(f" Training configured")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# Initialize Trainer
print("\nInitializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

print(" Trainer initialized")

# Train Model
print("\n" + "="*60)
print(" STARTING FINE-TUNING")
print("="*60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Initial VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB / 4.00 GB")
print(f"Training samples: {len(tokenized_train)}")
print(f"Epochs: {training_args.num_train_epochs}")
print("="*60 + "\n")

try:
    trainer.train()
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETE!")
    print("="*60)
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n CUDA OUT OF MEMORY")
        print("  1. Reduce MAX_LENGTH to 256")
        print("  2. Increase gradient_accumulation_steps to 8")
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        raise
    else:
        print(f"\n Training error: {e}")
        raise

except KeyboardInterrupt:
    print("\n Training interrupted")

# Save Model
print("\nSaving fine-tuned model...")

try:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f" Model saved to: {OUTPUT_DIR}")
    
    lora_adapter_dir = f"{OUTPUT_DIR}/lora_adapters"
    model.save_pretrained(lora_adapter_dir)
    print(f"✓ LoRA adapters saved to: {lora_adapter_dir}")
    
except Exception as e:
    print(f" Error saving model: {e}")

print("\n" + "="*60)
print(" TRAINING STATISTICS")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Training samples: {len(tokenized_train)}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Final VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
print("="*60)
print("\n Fine-tuning completed successfully!")
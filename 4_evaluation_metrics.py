import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from evaluate import load
from bert_score import score as bert_score
import numpy as np
from tqdm import tqdm
import pandas as pd
import re


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load Fine-Tuned Model
print("Loading fine-tuned model for evaluation...")
MODEL_DIR = "./llama3_disaster_lora"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # FIXED: Match training model

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    print(" Tokenizer loaded")
    
    # Load base model WITHOUT quantization (Windows compatible)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f" Base model loaded - VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, f"{MODEL_DIR}/lora_adapters")
    model.eval()
    print(f" LoRA adapters loaded - VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
except Exception as e:
    print(f" Error loading model: {e}")
    exit(1)

# Load Evaluation Metrics
print("\nLoading evaluation metrics...")
try:
    rouge = load('rouge')
    print(" ROUGE loaded")
    meteor = load('meteor')
    print(" METEOR loaded")
except Exception as e:
    print(f" Error loading metrics: {e}")
    print("Install with: pip install rouge-score nltk")
    exit(1)


# Load Test Data
print("\nLoading test data...")
try:
    with open('disaster_training_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use last 20% as test set (not used in training)
    test_size = int(len(data) * 0.2)
    test_data = data[-test_size:]
    print(f" Loaded {len(test_data)} test samples")
    
except FileNotFoundError:
    print(" Error: disaster_training_data.json not found!")
    exit(1)


# Generate Predictions
def generate_prediction(instruction: str) -> str:
    """Generate summary for given instruction"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert disaster response analyst. Provide concise, actionable summaries of disaster events.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|start_header_id|>assistant<|end_header_id|>" in generated:
        response = generated.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        response = generated
    
    # Clean up response
    response = re.split(r'<\|start_header_id\|>', response)[0].strip()
    response = re.split(r'<\|eot_id\|>', response)[0].strip()
    
    return response


print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)
print(f"Test samples: {len(test_data)}")
print(f"This may take several minutes...")
print("="*60 + "\n")

predictions = []
references = []

for i, sample in enumerate(tqdm(test_data, desc="Generating")):
    pred = generate_prediction(sample['instruction'])
    predictions.append(pred)
    references.append(sample['response'])
    
    # Show a sample every 50 iterations
    if (i + 1) % 50 == 0:
        print(f"\nSample prediction {i+1}:")
        print(f"  Instruction: {sample['instruction'][:80]}...")
        print(f"  Prediction: {pred[:100]}...")
        print()

print("\n✓ All predictions generated")

# ROUGE Evaluation
print("\n" + "="*60)
print("COMPUTING ROUGE SCORES")
print("="*60)

try:
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    print("\n ROUGE Scores:")
    print(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")
    
except Exception as e:
    print(f" ROUGE computation failed: {e}")
    rouge_results = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

# METEOR Evaluation
print("\n" + "="*60)
print("COMPUTING METEOR SCORE")
print("="*60)

try:
    meteor_scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="METEOR"):
        meteor_result = meteor.compute(predictions=[pred], references=[ref])
        meteor_scores.append(meteor_result['meteor'])
    
    meteor_avg = np.mean(meteor_scores)
    meteor_std = np.std(meteor_scores)
    print(f"\n METEOR Score: {meteor_avg:.4f} (±{meteor_std:.4f})")
    
except Exception as e:
    print(f" METEOR computation failed: {e}")
    meteor_avg = 0.0
    meteor_std = 0.0

# BERTScore Evaluation
print("\n" + "="*60)
print("COMPUTING BERTSCORE")
print("="*60)
print("This may take 5-10 minutes for large test sets...")

try:
    # Use a lighter model for faster computation on 4GB VRAM
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",  # Lighter than deberta-xlarge
        device=device,
        verbose=True,
        batch_size=8,  # Process in small batches
    )
    
    bert_score_f1 = F1.mean().item()
    bert_score_precision = P.mean().item()
    bert_score_recall = R.mean().item()
    
    print("\n BERTScore:")
    print(f"  Precision: {bert_score_precision:.4f}")
    print(f"  Recall: {bert_score_recall:.4f}")
    print(f"  F1: {bert_score_f1:.4f}")
    
except Exception as e:
    print(f" BERTScore computation failed: {e}")
    print("Skipping BERTScore (requires additional GPU memory)")
    bert_score_f1 = 0.0
    bert_score_precision = 0.0
    bert_score_recall = 0.0


# Save Results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results = {
    'model': BASE_MODEL,
    'test_samples': len(test_data),
    'metrics': {
        'rouge1': float(rouge_results['rouge1']),
        'rouge2': float(rouge_results['rouge2']),
        'rougeL': float(rouge_results['rougeL']),
        'meteor_mean': float(meteor_avg),
        'meteor_std': float(meteor_std),
        'bertscore_precision': float(bert_score_precision),
        'bertscore_recall': float(bert_score_recall),
        'bertscore_f1': float(bert_score_f1),
    }
}

with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(" Results saved to: evaluation_results.json")

# Save detailed samples
df = pd.DataFrame({
    'instruction': [s['instruction'][:150] for s in test_data],
    'reference': [r[:150] for r in references],
    'prediction': [p[:150] for p in predictions]
})
df.to_csv('evaluation_samples.csv', index=False)
print(" Samples saved to: evaluation_samples.csv")

# Save full predictions for manual review
full_df = pd.DataFrame({
    'instruction': [s['instruction'] for s in test_data],
    'reference': references,
    'prediction': predictions
})
full_df.to_csv('evaluation_full_predictions.csv', index=False)
print(" Full predictions saved to: evaluation_full_predictions.csv")


# Final Summary Report
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(f"Model: {BASE_MODEL}")
print(f"LoRA Adapters: {MODEL_DIR}/lora_adapters")
print(f"Test Samples: {len(test_data)}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
print("\n Evaluation Metrics:")
print(f"  ROUGE-1:      {rouge_results['rouge1']:.4f}")
print(f"  ROUGE-2:      {rouge_results['rouge2']:.4f}")
print(f"  ROUGE-L:      {rouge_results['rougeL']:.4f}")
print(f"  METEOR:       {meteor_avg:.4f} (±{meteor_std:.4f})")
if bert_score_f1 > 0:
    print(f"  BERTScore F1: {bert_score_f1:.4f}")
else:
    print(f"  BERTScore F1: Skipped (memory constraints)")

print("\n  Output Files:")
print("  - evaluation_results.json (metric summary)")
print("  - evaluation_samples.csv (preview)")
print("  - evaluation_full_predictions.csv (complete)")
print("="*80)
print("\n Evaluation complete!")

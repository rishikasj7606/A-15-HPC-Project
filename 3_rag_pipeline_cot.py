import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from typing import List, Dict
import re


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration
MODEL_DIR = "./llama3_disaster_lora"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Load Fine-Tuned Model
print("Loading fine-tuned LLaMA 3.2 1B model...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    print(" Tokenizer loaded")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f" Base model loaded - VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    model = PeftModel.from_pretrained(base_model, f"{MODEL_DIR}/lora_adapters")
    model.eval()
    print(f" LoRA adapters loaded - VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
except Exception as e:
    print(f" Error loading model: {e}")
    exit(1)

# Load RAG Components
print("\nLoading RAG components...")

try:
    embedding_model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device=device
    )
    print(" Embedding model loaded")
    
    index = faiss.read_index('disaster_faiss_index.bin')
    print(" FAISS index loaded (CPU)")
    
    with open('chunk_metadata.pkl', 'rb') as f:
        chunk_metadata = pickle.load(f)
    with open('text_chunks.pkl', 'rb') as f:
        text_chunks = pickle.load(f)
    print(f" Loaded {len(text_chunks)} text chunks")
    
except Exception as e:
    print(f" Error loading RAG components: {e}")
    exit(1)

# RAG Retrieval Function
def retrieve_relevant_docs(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve top-k relevant documents"""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(text_chunks):
            retrieved_docs.append({
                'text': text_chunks[idx],
                'metadata': chunk_metadata[idx],
                'score': float(distance)
            })
    
    return retrieved_docs

# Improved Generation Function
def generate_disaster_summary(query: str, max_length: int = 400) -> tuple:
    """
    Generate disaster summary using RAG
    Returns: (summary, retrieved_docs)
    """
    # Retrieve relevant documents
    print(f"  Retrieving documents...")
    retrieved_docs = retrieve_relevant_docs(query, top_k=3)
    
    # Format context (use full text from top documents)
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:3], 1):
        # Clean and format the document text
        doc_text = doc['text'].strip()
        context_parts.append(f"Document {i}: {doc_text[:400]}")
    
    context = "\n\n".join(context_parts)
    
    # Simplified prompt without confusing few-shot examples
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert disaster response analyst. Provide clear, concise summaries of disaster events based on the information provided.<|eot_id|><|start_header_id|>user<|end_header_id|>

Query: {query}

Context from recent disaster reports:
{context}

Based on the above information, provide a brief disaster analysis covering:
1. Disaster type and location
2. Severity and impact
3. Key response recommendations

Keep your response under 200 words.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Generate
    print(f"  Generating summary...")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1200
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response more reliably
    if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
        # Split and get everything after the last assistant marker
        parts = full_text.split("<|start_header_id|>assistant<|end_header_id|>")
        response = parts[-1].strip()
        
        # Remove any trailing system/user markers
        response = re.split(r'<\|start_header_id\|>', response)[0].strip()
        response = re.split(r'<\|eot_id\|>', response)[0].strip()
    else:
        response = full_text
    
    # Clean up response
    response = response.strip()
    
    # If response is too short or looks like it's repeating the prompt, provide fallback
    if len(response) < 50 or "system" in response[:100].lower():
        response = f"**Disaster Analysis for: {query}**\n\n"
        response += f"Based on {len(retrieved_docs)} retrieved documents:\n\n"
        
        # Extract key info from top document
        top_doc = retrieved_docs[0]['text']
        if 'Severity:' in top_doc:
            severity = re.search(r'Severity:\s*(\w+)', top_doc)
            if severity:
                response += f"- Severity Level: {severity.group(1).upper()}\n"
        
        if 'Location:' in top_doc:
            location = re.search(r'Location:\s*([^\n]+)', top_doc)
            if location and location.group(1).strip() != "None":
                response += f"- Location: {location.group(1)}\n"
        
        response += f"\nSummary: {top_doc[:200]}..."
    
    return response, retrieved_docs

print("\n" + "="*80)
print("DISASTER RESPONSE RAG SYSTEM - READY")
print("="*80)
print(f"Model: {BASE_MODEL}")
print(f"Device: {device}")
print(f"VRAM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Knowledge Base: {len(text_chunks)} disaster reports")
print("="*80)
print("\nSystem ready. Use generate_disaster_summary(query) to generate responses.")
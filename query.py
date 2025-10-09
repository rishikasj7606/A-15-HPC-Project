import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re


class FactBasedDisasterSystem:
    
    def __init__(self, model_dir="./llama3_disaster_lora", base_model="meta-llama/Llama-3.2-1B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing system on {self.device}...")
        
        # Only load for embeddings (not generation)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        self.model = PeftModel.from_pretrained(base, f"{model_dir}/lora_adapters")
        self.model.eval()
        
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=self.device
        )
        
        self.index = faiss.read_index('disaster_faiss_index.bin')
        
        with open('chunk_metadata.pkl', 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        
        with open('text_chunks.pkl', 'rb') as f:
            self.text_chunks = pickle.load(f)
        
        print(f" Ready - {len(self.text_chunks)} reports loaded\n")
    
    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve relevant documents"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k * 2)
        
        docs = []
        seen = set()
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                title = self.chunk_metadata[idx].get('title', '')
                if title and title not in seen:
                    seen.add(title)
                    docs.append({
                        'text': self.text_chunks[idx],
                        'metadata': self.chunk_metadata[idx],
                        'score': float(dist)
                    })
                if len(docs) >= top_k:
                    break
        
        return docs
    
    def extract_facts(self, text: str) -> dict:
        """Extract all facts from document text"""
        facts = {}
        
        # Title
        m = re.search(r'Title:\s*([^\n]+)', text)
        facts['title'] = m.group(1).strip() if m else None
        
        # Deaths
        m = re.search(r'(\d+)\s+deaths?', text, re.I)
        facts['deaths'] = int(m.group(1)) if m else None
        
        # Displaced
        m = re.search(r'(\d+)\s+displaced', text, re.I)
        facts['displaced'] = int(m.group(1)) if m else None
        
        # Dates
        m = re.search(r'On\s+(\d{2}/\d{2}/\d{4}).*?until\s+(\d{2}/\d{2}/\d{4})', text)
        if m:
            facts['date_start'] = m.group(1)
            facts['date_end'] = m.group(2)
        else:
            m = re.search(r'(?:On|started).*?(\d{2}/\d{2}/\d{4})', text, re.I)
            facts['date_start'] = m.group(1) if m else None
            facts['date_end'] = None
        
        # Severity
        m = re.search(r'Severity:\s*(\w+)', text, re.I)
        facts['severity'] = m.group(1).lower() if m else None
        
        # Type
        m = re.search(r'Type:\s*(\w+)', text, re.I)
        facts['type'] = m.group(1).lower() if m else None
        
        # Location from title
        if facts['title']:
            m = re.search(r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', facts['title'])
            facts['location'] = m.group(1) if m else None
        else:
            facts['location'] = None
        
        return facts
    
    def answer_question(self, query: str, docs: list) -> str:
        """Answer question using extracted facts only"""
        
        if not docs:
            return "No information found for this query."
        
        # Extract facts from top document
        facts = self.extract_facts(docs[0]['text'])
        query_lower = query.lower()
        
        # Question type detection and answering
        
        # HOW MANY DIED
        if any(w in query_lower for w in ['how many', 'dead', 'death', 'died', 'killed', 'casualties']):
            if facts['deaths'] is not None:
                answer = f"{facts['deaths']} deaths were reported"
                if facts['location']:
                    answer += f" in the {facts['location']}"
                answer += f" {facts['type'] or 'disaster'}"
                if facts['displaced']:
                    answer += f".\n\nAdditionally, {facts['displaced']} people were displaced."
                return answer
            return "Death toll information not available for this event."
        
        # WHEN / DATE
        if any(w in query_lower for w in ['when', 'date', 'happen', 'occur', 'start']):
            if facts['date_start']:
                answer = f"{facts['title'] or 'The event'}\n\n"
                answer += f" Started: {facts['date_start']}"
                if facts['date_end']:
                    answer += f"\nEnded: {facts['date_end']}"
                if facts['severity']:
                    answer += f"\n Severity: {facts['severity'].upper()}"
                return answer
            return "Date information not available for this event."
        
        # WHERE / LOCATION
        if any(w in query_lower for w in ['where', 'location', 'place', 'region']):
            if facts['location']:
                answer = f"{facts['title'] or 'Event'}\n\n"
                answer += f" Location: {facts['location']}\n"
                answer += f" Severity: {facts['severity'].upper() if facts['severity'] else 'Unknown'}"
                return answer
            return f"Location not specified for: {facts['title'] or 'this event'}"
        
        # SEVERITY
        if any(w in query_lower for w in ['severity', 'how severe', 'how bad']):
            if facts['severity']:
                answer = f"{facts['title'] or 'Event'}\n\n"
                answer += f" Severity: {facts['severity'].upper()}\n"
                if facts['deaths'] or facts['displaced']:
                    answer += f"\nImpact:\n"
                    if facts['deaths']:
                        answer += f"- Deaths: {facts['deaths']}\n"
                    if facts['displaced']:
                        answer += f"- Displaced: {facts['displaced']}"
                return answer
            return "Severity information not available."
        
        # GENERAL SUMMARY
        answer = f"**{facts['title'] or 'Disaster Event'}**\n\n"
        
        if facts['type']:
            answer += f"Type: {facts['type'].upper()}\n"
        if facts['severity']:
            answer += f"Severity: {facts['severity'].upper()}\n"
        if facts['location']:
            answer += f"Location: {facts['location']}\n"
        
        if facts['date_start']:
            answer += f"\n Timeline: {facts['date_start']}"
            if facts['date_end']:
                answer += f" to {facts['date_end']}"
            answer += "\n"
        
        if facts['deaths'] or facts['displaced']:
            answer += f"\n Impact:\n"
            if facts['deaths']:
                answer += f"- Deaths: {facts['deaths']}\n"
            if facts['displaced']:
                answer += f"- Displaced: {facts['displaced']}\n"
        
        return answer.strip()
    
    def query(self, user_query: str):
        """Main query method"""
        
        docs = self.retrieve(user_query, top_k=3)
        
        if not docs:
            return {
                'query': user_query,
                'response': "No relevant disaster information found.",
                'sources': []
            }
        
        response = self.answer_question(user_query, docs)
        
        sources = [{
            'title': d['metadata'].get('title', 'Unknown')[:60],
            'type': d['metadata'].get('disaster_type', 'unknown'),
            'severity': d['metadata'].get('severity', 'unknown'),
            'relevance': f"{d['score']:.3f}"
        } for d in docs]
        
        return {
            'query': user_query,
            'response': response,
            'sources': sources
        }


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print(" STARTING DISASTER RESPONSE SYSTEM")
    print("="*80 + "\n")
    
    system = FactBasedDisasterSystem()
    
    # Test queries
    tests = [
        "How many people died in Indonesia flood?",
        "When did Nepal flood happen?",
        "Where was the recent earthquake?",
        "What is the severity of Malaysia flood?",
        "Tell me about recent earthquakes"
    ]
    
    print("Running test queries...\n")
    
    for query in tests:
        result = system.query(query)
        print(f" {result['query']}")
        print("-"*80)
        print(result['response'])
        print(f"\n Sources: {len(result['sources'])} documents")
        for i, src in enumerate(result['sources'], 1):
            print(f"  {i}. {src['title']} | {src['type']} | {src['severity']}")
        print("\n" + "="*80 + "\n")
    
    # Interactive mode
    print("Enter your queries (type 'quit' to exit):\n")
    
    while True:
        try:
            user_input = input("Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n System shutdown.")
                break
            
            if not user_input:
                continue
            
            result = system.query(user_input)
            
            print("\n" + "="*80)
            print(result['response'])
            print("\n Sources:")
            for i, src in enumerate(result['sources'], 1):
                print(f"  {i}. {src['title']} | Relevance: {src['relevance']}")
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nSystem shutdown.")
            break
        except Exception as e:
            print(f"\n Error: {e}")
    
    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

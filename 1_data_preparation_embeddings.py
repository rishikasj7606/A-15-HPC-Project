import torch
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle
from tqdm import tqdm
from datetime import datetime
import re


# Check CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_db"]
collection = db["events"]


# Load Embedding Model
print("Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
embedding_dim = 384


# Fetch Data from MongoDB
print("Fetching data from MongoDB...")
documents = list(collection.find({}))
print(f"Total documents retrieved: {len(documents)}")


if len(documents) == 0:
    print("No documents found in MongoDB. Please run the data collection script first.")
    exit(1)

# IMPROVED: Filter Real Disasters
def is_real_disaster(doc):
    """Filter out non-disaster content like movies, studies, political articles"""
    title = doc.get('title', '').lower()
    summary = doc.get('summary', '').lower()
    
    # Exclude entertainment content
    if any(keyword in title or keyword in summary for keyword in [
        'movie', 'film', 'trailer', 'sequel', 'actor', 'actress', 
        'premiere', 'box office', 'director'
    ]):
        return False
    
    # Exclude political/fundraising content
    if any(keyword in title for keyword in [
        'trump', 'biden', 'campaign', 'fundraising', 'political',
        'election', 'presidential'
    ]):
        return False
    
    # Exclude studies/research unless they report actual events
    if 'study finds' in title or 'research shows' in title:
        if not any(keyword in summary for keyword in ['occurred', 'reported', 'affected']):
            return False
    
    # Exclude cyber/business incidents unless catastrophic
    if 'cyberattack' in title.lower() or 'cyber attack' in title.lower():
        # Only include if it mentions major infrastructure/utilities impact
        if not any(keyword in summary.lower() for keyword in [
            'power grid', 'hospital', 'emergency services', 'critical infrastructure'
        ]):
            return False
    
    return True


# Filter documents
print("Filtering real disaster events...")
real_disasters = [doc for doc in documents if is_real_disaster(doc)]
print(f"Real disaster events: {len(real_disasters)} (filtered {len(documents) - len(real_disasters)} non-disasters)")

# Disaster Type Detection
def detect_disaster_type(doc):
    """Detect specific disaster type from document"""
    title = doc.get('title', '').lower()
    summary = doc.get('summary', '').lower()
    text = title + ' ' + summary
    
    disaster_types = {
        'earthquake': ['earthquake', 'seismic', 'tremor', 'magnitude', 'epicenter', 'aftershock'],
        'flood': ['flood', 'flooding', 'heavy rain', 'monsoon', 'deluge', 'inundation'],
        'wildfire': ['wildfire', 'forest fire', 'bushfire', 'blaze', 'fire'],
        'hurricane': ['hurricane', 'cyclone', 'typhoon', 'tropical storm'],
        'tornado': ['tornado', 'twister'],
        'volcano': ['volcano', 'volcanic', 'eruption', 'lava'],
        'tsunami': ['tsunami', 'tidal wave'],
        'drought': ['drought', 'water shortage', 'dry spell'],
        'landslide': ['landslide', 'mudslide', 'rockslide'],
        'avalanche': ['avalanche'],
        'heatwave': ['heat wave', 'heatwave', 'extreme heat'],
        'winter_storm': ['blizzard', 'winter storm', 'snowstorm', 'ice storm']
    }
    
    for dtype, keywords in disaster_types.items():
        if any(keyword in text for keyword in keywords):
            return dtype
    
    return 'general'


# IMPROVED: Event-Specific Recommendations
def get_specific_recommendations(disaster_type, severity):
    """Generate disaster-specific recommendations"""
    
    base_actions = {
        'earthquake': [
            "Drop, Cover, and Hold On if shaking occurs",
            "Check for injuries and provide first aid if trained",
            "Inspect building for cracks, gas leaks, and structural damage",
            "Be prepared for aftershocks - do not enter damaged structures",
            "Turn off utilities if damage is suspected"
        ],
        'flood': [
            "Evacuate to higher ground immediately if ordered",
            "Never walk or drive through flood waters (6 inches can knock you down)",
            "Avoid contact with flood water - may be contaminated",
            "Move valuables to upper floors if time permits",
            "Do not return home until authorities declare it safe"
        ],
        'wildfire': [
            "Evacuate immediately if ordered - do not delay",
            "Close all windows, vents, and doors to prevent ember entry",
            "Remove flammable materials from around your home",
            "Wear N95 masks if exposed to smoke",
            "Monitor air quality and evacuation routes continuously"
        ],
        'hurricane': [
            "Evacuate coastal areas if in mandatory evacuation zone",
            "Stock emergency supplies: water (1 gal/person/day for 3 days), non-perishable food",
            "Secure outdoor items and board up windows",
            "Identify safe room away from windows",
            "Stay indoors during the storm - do not go outside during the eye"
        ],
        'tornado': [
            "Seek shelter in basement or interior room on lowest floor",
            "Avoid windows - get under sturdy furniture if possible",
            "If in mobile home, evacuate to sturdy building immediately",
            "If caught outside, lie flat in a ditch and cover head",
            "Monitor weather radio for warnings"
        ],
        'general': [
            "Monitor official emergency channels for updates",
            "Follow local emergency management directives",
            "Ensure emergency kit is accessible (water, food, first aid, flashlight)",
            "Know evacuation routes and have transportation plan",
            "Stay informed and help vulnerable neighbors if safe to do so"
        ]
    }
    
    actions = base_actions.get(disaster_type, base_actions['general'])
    
    # Add severity-specific actions
    if severity in ['high', 'critical', 'severe']:
        actions.insert(0, "HIGH SEVERITY - Treat as life-threatening emergency")
    
    return actions


def create_improved_response(doc, disaster_type):
    """Create contextual, non-templated response"""
    
    title = doc.get('title', 'Disaster Event')
    summary = doc.get('summary', '')
    location = doc.get('location')
    severity = doc.get('severity', 'unknown')
    date = doc.get('date', 'Unknown date')
    
    # Format location
    location_text = location if location and location != 'None' else "location to be confirmed"
    
    # Extract casualties if mentioned
    casualties = ""
    if summary:
        death_match = re.search(r'(\d+)\s+(?:deaths?|killed|dead|fatalities)', summary, re.IGNORECASE)
        injured_match = re.search(r'(\d+)\s+(?:injured|wounded)', summary, re.IGNORECASE)
        displaced_match = re.search(r'(\d+)\s+(?:displaced|evacuated|homeless)', summary, re.IGNORECASE)
        
        if death_match or injured_match or displaced_match:
            casualties = "\n\n**Casualties & Impact:**\n"
            if death_match:
                casualties += f"- Deaths: {death_match.group(1)}\n"
            if injured_match:
                casualties += f"- Injured: {injured_match.group(1)}\n"
            if displaced_match:
                casualties += f"- Displaced: {displaced_match.group(1)}\n"
    
    # Get specific recommendations
    recommendations = get_specific_recommendations(disaster_type, severity)
    rec_text = "\n".join([f"- {action}" for action in recommendations])
    
    # Generate varied response format
    response_templates = [
        # Format 1: Structured
        f"""**{disaster_type.replace('_', ' ').title()} Event**

**Event:** {title}
**Location:** {location_text}
**Severity:** {severity.upper()}
**Date:** {date}

**Summary:** {summary}{casualties}

**Immediate Actions:**
{rec_text}""",
        
        # Format 2: Narrative
        f"""A {severity}-severity {disaster_type.replace('_', ' ')} event has been reported in {location_text} on {date}.

{summary}

**Critical Response Actions:**
{rec_text}""",
        
        # Format 3: Concise
        f"""{title}

Severity: {severity.upper()} | Location: {location_text} | Date: {date}

{summary[:300]}{"..." if len(summary) > 300 else ""}{casualties}

**Response Protocol:**
{rec_text}"""
    ]
    
    # Select format based on document index (for variety)
    format_idx = hash(str(doc.get('_id', ''))) % len(response_templates)
    return response_templates[format_idx]

# IMPROVED: Prepare Training Data
def prepare_improved_training_data(docs):
    """Prepare high-quality training data"""
    training_data = []
    
    for doc in tqdm(docs, desc="Preparing training data"):
        disaster_type = detect_disaster_type(doc)
        
        # Create varied instruction formats
        instruction_formats = [
            f"Provide an emergency response summary for the following {disaster_type.replace('_', ' ')} event",
            f"Analyze this {disaster_type.replace('_', ' ')} disaster and recommend immediate actions",
            f"Summarize the following disaster event with specific safety recommendations"
        ]
        
        # Build context
        context = f"""Source: {doc.get('source', 'Unknown')}
Title: {doc.get('title', 'No title')}
Date: {doc.get('date', 'Unknown')}
Location: {doc.get('location', 'Location TBD')}
Severity: {doc.get('severity', 'Unknown')}
Summary: {doc.get('summary', 'No summary')}"""
        
        # Select instruction format
        instruction_idx = hash(str(doc.get('_id', ''))) % len(instruction_formats)
        instruction = f"{instruction_formats[instruction_idx]}:\n\n{context}"
        
        # Create response
        response = create_improved_response(doc, disaster_type)
        
        training_data.append({
            "instruction": instruction,
            "response": response,
            "metadata": {
                "source": doc.get('source'),
                "title": doc.get('title'),
                "date": str(doc.get('date')),
                "location": doc.get('location'),
                "severity": doc.get('severity'),
                "disaster_type": disaster_type
            }
        })
    
    return training_data


print("\nGenerating improved training data...")
training_data = prepare_improved_training_data(real_disasters)
print(f"Prepared {len(training_data)} high-quality training examples")


# Save training data
with open('disaster_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print("Training data saved to: disaster_training_data.json")

# RAG: Create Document Chunks
def create_chunks(docs, chunk_size=300):
    """Create text chunks for embedding"""
    chunks = []
    metadata = []
    
    for idx, doc in enumerate(docs):
        # Only include real disaster information
        disaster_type = detect_disaster_type(doc)
        
        text = f"Title: {doc.get('title', '')} Summary: {doc.get('summary', '')} Location: {doc.get('location', '')} Severity: {doc.get('severity', '')} Type: {disaster_type}"
        
        # Chunk by words
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                metadata.append({
                    'doc_id': str(doc.get('_id')),
                    'source': doc.get('source'),
                    'title': doc.get('title'),
                    'disaster_type': disaster_type,
                    'severity': doc.get('severity'),
                    'chunk_idx': i // chunk_size
                })
    
    return chunks, metadata


print("\nCreating document chunks for RAG...")
text_chunks, chunk_metadata = create_chunks(real_disasters)
print(f"Created {len(text_chunks)} chunks")


# Generate Embeddings
print("Generating embeddings...")
batch_size = 32
embeddings_list = []

for i in tqdm(range(0, len(text_chunks), batch_size), desc="Encoding"):
    batch = text_chunks[i:i+batch_size]
    batch_embeddings = embedding_model.encode(
        batch,
        convert_to_numpy=True,
        show_progress_bar=False,
        device=device,
        batch_size=batch_size
    )
    embeddings_list.append(batch_embeddings)

embeddings = np.vstack(embeddings_list).astype('float32')
print(f"Embeddings shape: {embeddings.shape}")


# Create FAISS Index
print("Creating FAISS index...")
index = faiss.IndexFlatIP(embedding_dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"Added {index.ntotal} vectors to FAISS index")


# Save everything
print("\nSaving files...")
faiss.write_index(index, 'disaster_faiss_index.bin')

with open('chunk_metadata.pkl', 'wb') as f:
    pickle.dump(chunk_metadata, f)

with open('text_chunks.pkl', 'wb') as f:
    pickle.dump(text_chunks, f)


# Test
print("\n" + "="*70)
print("TESTING FAISS INDEX")
print("="*70)

test_queries = [
    "earthquake high severity casualties",
    "flood coastal areas evacuation",
    "wildfire emergency response"
]

for test_query in test_queries:
    print(f"\nQuery: '{test_query}'")
    test_embedding = embedding_model.encode([test_query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(test_embedding)
    
    distances, indices = index.search(test_embedding, 3)
    
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        print(f"  {i}. Score: {dist:.3f} | Type: {chunk_metadata[idx].get('disaster_type', 'unknown')}")
        print(f"     {chunk_metadata[idx]['title'][:70]}...")

print("\n" + "="*70)
print(" DATA PREPARATION COMPLETE")
print("="*70)
print(f"Original documents:      {len(documents)}")
print(f"Real disaster events:    {len(real_disasters)}")
print(f"Training examples:       {len(training_data)}")
print(f"Text chunks:             {len(text_chunks)}")
print(f"Embeddings:              {embeddings.shape}")
print(f"\nFiles created:")
print(f"  disaster_training_data.json ({len(training_data)} samples)")
print(f"  disaster_faiss_index.bin ({index.ntotal} vectors)")
print(f"  chunk_metadata.pkl")
print(f"  text_chunks.pkl")
print(f"\nNext step: Review disaster_training_data.json to verify quality")
print("="*70)

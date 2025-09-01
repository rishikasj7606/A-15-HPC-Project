##Real-Time Disaster Response Summarization Using LLM


!pip install transformers datasets accelerate peft bitsandbytes
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install requests beautifulsoup4 pandas numpy matplotlib seaborn
!pip install evaluate rouge-score bert-score sacrebleu

import os
import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')
import torch

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    pipeline, BitsAndBytesConfig
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate
from bert_score import score


class CrisisDataCollector:
    def __init__(self):
        self.news_api_key = "6cfb3ad6433845bea794f002fa1913e3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def collect_reliefweb_data(self, limit=50):
        """Collect crisis reports from ReliefWeb API"""
        url = "https://api.reliefweb.int/v1/reports"
        params = {
            'appname': 'crisis-llm',
            'limit': 50,
            'filter[status]': 'published',
            'filter[date.created][from]': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'fields[include]': ['title', 'body', 'date', 'disaster_type', 'country', 'source']
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            reports = []
            for item in data.get('data', []):
                fields = item.get('fields', {})
                reports.append({
                    'title': fields.get('title', ''),
                    'body': fields.get('body', ''),
                    'date': fields.get('date', {}).get('created', ''),
                    'disaster_type': fields.get('disaster_type', []),
                    'country': fields.get('country', []),
                    'source': fields.get('source', []),
                    'data_source': 'reliefweb'
                })

            print(f"Collected {len(reports)} reports from ReliefWeb")
            return reports

        except Exception as e:
            print(f"Error collecting ReliefWeb data: {e}")
            return []

    def collect_gdacs_data(self):
        """Collect disaster alerts from GDACS"""
        url = "https://www.gdacs.org/xml/rss.xml"

        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()

            # Parse RSS/XML (simplified parsing)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            alerts = []
            for item in root.findall('.//item'):
                title = item.find('title')
                description = item.find('description')
                pub_date = item.find('pubDate')

                alerts.append({
                    'title': title.text if title is not None else '',
                    'body': description.text if description is not None else '',
                    'date': pub_date.text if pub_date is not None else '',
                    'data_source': 'gdacs'
                })

            print(f"Collected {len(alerts)} alerts from GDACS")
            return alerts

        except Exception as e:
            print(f"Error collecting GDACS data: {e}")
            return []

    def collect_usgs_data(self):
        """Collect earthquake data from USGS"""
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            earthquakes = []
            for feature in data.get('features', []):
                props = feature.get('properties', {})
                coords = feature.get('geometry', {}).get('coordinates', [])

                # Generate descriptive text
                magnitude = props.get('mag', 0)
                place = props.get('place', 'Unknown location')
                time_ms = props.get('time', 0)
                date = datetime.fromtimestamp(time_ms/1000).strftime('%Y-%m-%d %H:%M:%S')

                body = f"Magnitude {magnitude} earthquake occurred at {place}. "
                body += f"Depth: {coords[2] if len(coords) > 2 else 'Unknown'} km. "
                body += f"Occurred on {date}."

                earthquakes.append({
                    'title': f"M{magnitude} - {place}",
                    'body': body,
                    'date': date,
                    'magnitude': magnitude,
                    'coordinates': coords,
                    'data_source': 'usgs'
                })

            print(f"Collected {len(earthquakes)} earthquakes from USGS")
            return earthquakes

        except Exception as e:
            print(f"Error collecting USGS data: {e}")
            return []

    def collect_news_data(self, query="earthquake OR flood OR hurricane OR disaster", limit=50):
        """Collect news articles using News API"""
        if not self.news_api_key or self.news_api_key == "YOUR_NEWS_API_KEY":
            print("News API key not provided. Skipping news collection.")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            'apiKey': self.news_api_key,
            'q': query,
            'sortBy': 'publishedAt',
            'pageSize': min(limit, 100),
            'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'body': article.get('description', '') + ' ' + article.get('content', ''),
                    'date': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ' '),
                    'data_source': 'news_api'
                })

            print(f"Collected {len(articles)} news articles")
            return articles

        except Exception as e:
            print(f"Error collecting news data: {e}")
            return []

    def collect_all_data(self):
        """Collect data from all sources"""
        all_data = []

        print("Collecting crisis data from multiple sources...")

        # Collect from all sources
        all_data.extend(self.collect_reliefweb_data())
        all_data.extend(self.collect_gdacs_data())
        all_data.extend(self.collect_usgs_data())
        all_data.extend(self.collect_news_data())

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Clean and preprocess
        if not df.empty:
            df['body'] = df['body'].fillna('')
            df['title'] = df['title'].fillna('')
            df['body'] = df['body'].str.strip()
            df['title'] = df['title'].str.strip()
            nbh
            # Remove empty entries
            df = df[(df['title'] != '') | (df['body'] != '')]

        print(f"Total collected: {len(df)} crisis reports")
        return df

# Assume df is the DataFrame from CrisisDataCollector.collect_all_data()
# We'll create input: "Summarize this crisis report" and target: the body text

def prepare_dataset(df: pd.DataFrame):
    # Make sure 'body' is not empty
    df = df[df['body'].str.strip() != ''].reset_index(drop=True)

    dataset = pd.DataFrame({
        "input_text": ["Summarize this crisis report: " + t for t in df['title']],
        "target_text": df['body']
    })

    from datasets import Dataset
    return Dataset.from_pandas(dataset)

dataset = prepare_dataset(df)


model_name = "t5-small"  # You can replace with a larger model if GPU memory allows
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_input_length = 512
max_target_length = 256

def tokenize(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=max_input_length, truncation=True)
    labels = tokenizer(batch["target_text"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)


from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer
# Load base model in 4-bit
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],  # typical target modules
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)


#Prepare the HuggingFace Dataset
from datasets import Dataset
import pandas as pd

# Example: df is your DataFrame from CrisisDataCollector
df = pd.DataFrame({
    "title": ["Earthquake in Tokyo", "Flood in Germany"],
    "body": ["Magnitude 6.5 earthquake hit Tokyo causing damage.",
             "Severe flooding in Germany affected thousands of people."]
})

dataset = pd.DataFrame({
    "input_text": ["Summarize this crisis report: " + t for t in df['title']],
    "target_text": df['body']
})

dataset = Dataset.from_pandas(dataset)

# Tokenize
from transformers import AutoTokenizer

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_input_length = 512
max_target_length = 256

def tokenize(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=max_input_length, truncation=True)
    labels = tokenizer(batch["target_text"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize, batched=True)


from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Load model and apply QLoRA
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./crisis_llm_qora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    fp16=True,
    optim="paged_adamw_32bit",
    do_eval=False,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator
)


model_name = "t5-small"  # You can replace with a larger model if GPU memory allows
tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer.train()
trainer.save_model("./crisis_llm_qora")
from transformers import pipeline

summarizer = pipeline(
    "text2text-generation",
    model="./crisis_llm_qora",
    tokenizer=tokenizer,
    device=0  # GPU
)

new_reports = [
    "Earthquake of magnitude 6.5 struck near Tokyo causing significant damage."
]

for report in new_reports:
    summary = summarizer("Summarize this crisis report: " + report, max_length=150, clean_up_tokenization_spaces=True)
    print(summary[0]['generated_text'])


# Create an instance of the data collector and collect data
data_collector = CrisisDataCollector()
df = data_collector.collect_all_data()

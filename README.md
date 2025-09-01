# A15 Real-Time Disaster Response Summarization using LLM with HPC

## Team Members
- **Harikrishna S** - CB.AI.U4AID23011  
- **Deepak Kumar** - CB.AI.U4AID23012  
- **Somavaram J Rishika** - CB.AI.U4AID23038  

## Overview
A high-performance computing (HPC)-enabled, AI-powered system designed to:
- Ingest real-time disaster data from multiple APIs in parallel  
- Normalize and store events in MongoDB with fast query indexing  
- Fine-tune LLM models for disaster classification & summarization  
- Deploy a Retrieval-Augmented Generation (RAG) pipeline for context-aware insights  
- Scale seamlessly using Docker & cloud-native architectures  

## Tech Stack
- **Languages:** Python, Bash  
- **Frameworks:** asyncio, aiohttp, Pydantic, HuggingFace Transformers  
- **Database:** MongoDB (with indexing)  
- **Parallel Computing:** MPI, Multiprocessing, AsyncIO  
- **Containerization:** Docker, Docker Compose  
- **Cloud:** AWS ECS/EKS, GCP Cloud Run, Azure Container Apps  

## Key Features
- ⚡ **Parallel API Ingestion** → Low-latency disaster data collection  
- 🧹 **Data Normalization & Validation** → Unified schema with Pydantic  
- 🗄 **MongoDB Storage** → Indexed for high-speed retrieval  
- 🤖 **LLM-based Summarization** → Context-aware disaster reports  
- 🔍 **RAG Pipeline** → Combines retrieval & generative AI for real-time insights  
- 🐳 **Dockerized Deployment** → Cloud-ready with CI/CD integration  

## Workflow
API Ingestion (AsyncIO)
→
Data Normalization & Validation (Pydantic)
→
MongoDB Storage with Indexing
→
Model Fine-Tuning & Preprocessing (Multiprocessing + HPC)
→
RAG Pipeline (Retrieval + LLM Summarization)
→
Docker + Cloud Deployment

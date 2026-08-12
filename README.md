# ActumAI Core Engine 🚀

Proprietary enterprise AI engine developed by **ActumAI sp. z o.o.** providing high-throughput RAG orchestration, LLM security guardrails, and autonomous B2B agentic workflows.

## 🌟 Core Modules

### 1. Enterprise Legal RAG (`/modules/legal_rag`)
* **Objective:** High-precision Retrieval-Augmented Generation system for Polish Statutory Law (Kodeks Cywilny, KSH, RODO).
* **Key Features:** Custom hierarchical chunking, vector embeddings via Qdrant/Chroma, and re-ranking pipeline to eliminate model hallucinations in legal advisory.

### 2. LLM Guardrail & Red-Teaming Proxy (`/modules/guardrail_proxy`)
* **Objective:** Security middleware intercepting user prompts prior to LLM execution.
* **Key Features:** Protection against Prompt Injections, Jailbreaking techniques, PII data leakage, and system prompt extraction.

### 3. Autonomous B2B Sales Agent (`/modules/crm_agent`)
* **Objective:** Multi-tool autonomous agent driving automated Discovery & Lead Qualification.
* **Key Features:** Native Function/Tool Calling, CRM integration via REST APIs, dynamic budget estimation.

## 🛠 Tech Stack
* **Language:** Python 3.11+
* **Frameworks:** FastAPI, LlamaIndex, LangChain / LangGraph
* **Vector DB:** Qdrant / ChromaDB
* **DevOps:** Docker, Docker-Compose, REST API


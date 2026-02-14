# ForgeLLM 🔥

**Fine-tune LLMs with LoRA - A SaaS Platform for LLM Fine-tuning**

ForgeLLM is a production-ready platform for fine-tuning Large Language Models using Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA. Built with a microservices architecture, it provides multi-tenant workspaces, async training jobs, and adapter-based inference.

## 🏗 Architecture

```
forgellm/
├── backend/          # FastAPI REST API
├── ml/               # ML training & inference
├── workers/          # Celery async workers
├── frontend/         # (Coming soon) React/Next.js UI
├── infra/            # Docker & deployment configs
└── docker-compose.yml
```

## ✨ Features

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with QLoRA support
- **Multi-tenant Workspaces**: Isolated environments for different projects
- **Async Training Jobs**: Non-blocking training with Celery workers
- **Adapter Management**: Hot-swap adapters for different tasks
- **REST API**: Full API for datasets, training, and inference
- **GPU Optimized**: 4-bit quantization for efficient memory usage

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA 11.8+ (for training)
- 16GB+ GPU VRAM recommended

### 1. Clone and Setup

```bash
git clone https://github.com/AngadGarcha0301/ForgeLLM.git
cd ForgeLLM

# Create environment file
cp .env.example .env
```

### 2. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 3. Or Run Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt

# Start PostgreSQL and Redis (via Docker)
docker-compose up -d postgres redis

# Run migrations
# alembic upgrade head  # (after setting up alembic)

# Start backend
uvicorn backend.app.main:app --reload

# In another terminal - Start Celery worker
celery -A workers.celery_app worker --loglevel=info
```

### 4. Access the API

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## 📖 API Usage

### Register & Login

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "username": "user", "password": "password123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=user@example.com&password=password123"
```

### Upload Dataset

```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "workspace_id=1" \
  -F "file=@your_dataset.jsonl"
```

Dataset format (JSONL):
```json
{"instruction": "Summarize the text", "input": "Long text here...", "output": "Summary here"}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

### Start Training

```bash
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": 1,
    "dataset_id": 1,
    "name": "my-first-finetune",
    "base_model": "mistralai/Mistral-7B-v0.1",
    "config": {
      "lora_r": 16,
      "lora_alpha": 32,
      "num_epochs": 3,
      "batch_size": 4,
      "learning_rate": 2e-4
    }
  }'
```

### Check Training Status

```bash
curl http://localhost:8000/api/v1/training/1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Run Inference

```bash
curl -X POST http://localhost:8000/api/v1/inference/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "prompt": "### Instruction:\nSummarize the following text\n\n### Input:\nYour text here...\n\n### Response:\n",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## 🔧 Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank (8, 16, 32, 64) |
| `lora_alpha` | 32 | LoRA alpha (typically 2x rank) |
| `lora_dropout` | 0.05 | Dropout probability |
| `learning_rate` | 2e-4 | Learning rate |
| `num_epochs` | 3 | Number of training epochs |
| `batch_size` | 4 | Batch size per device |
| `max_steps` | -1 | Max steps (-1 for full epochs) |

### Supported Base Models

- `mistralai/Mistral-7B-v0.1`
- `meta-llama/Llama-2-7b-hf` (requires HF token)
- `microsoft/phi-2`
- Any HuggingFace causal LM model

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── dependencies.py      # DI
│   ├── api/                 # Route handlers
│   ├── core/                # Business logic
│   ├── db/                  # Database models
│   ├── services/            # Service layer
│   └── utils/               # Utilities

ml/
├── preprocessing/           # Data formatting
├── training/                # LoRA training
├── evaluation/              # Metrics
└── inference/               # Model loading

workers/
├── celery_app.py           # Celery config
└── tasks.py                # Async tasks
```

## 🛣 Roadmap

### Phase 1 (MVP) ✅
- [x] FastAPI backend
- [x] Database models
- [x] Dataset upload
- [x] LoRA training pipeline
- [x] Celery workers
- [x] Basic inference

### Phase 2 (In Progress)
- [ ] Frontend UI (Next.js)
- [ ] Database migrations (Alembic)
- [ ] Training progress streaming
- [ ] Model comparison

### Phase 3
- [ ] Multi-GPU support
- [ ] Kubernetes deployment
- [ ] Model versioning
- [ ] A/B testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Celery](https://celeryproject.org/)
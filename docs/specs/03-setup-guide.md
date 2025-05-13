# RAG System Setup Guide

## Local Development Setup

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd rag
```

### Step 2: Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Unix/macOS
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Environment Variables
Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
ELASTICSEARCH_HOST=http://localhost:9200
```

### Step 4: Start Services
```bash
# Start Elasticsearch
docker-compose up -d elasticsearch

# Wait for Elasticsearch to be ready
curl http://localhost:9200
```

### Step 5: Run Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src
```

## Docker Deployment

### Step 1: Build Image
```bash
docker-compose build
```

### Step 2: Start Services
```bash
docker-compose up -d
```

### Step 3: Verify
```bash
# Check services
docker-compose ps

# Check logs
docker-compose logs -f
```

## Configuration

### Elasticsearch Settings
```yaml
# config/elasticsearch.yml
cluster.name: rag-cluster
node.name: rag-node
network.host: 0.0.0.0
discovery.type: single-node
xpack.security.enabled: false
```

### Application Settings
```python
# config/settings.py
config = {
    'elasticsearch': {
        'host': 'http://localhost',
        'port': 9200,
        'index': 'documents'
    },
    'chunking': {
        'chunk_size': 1000,
        'chunk_overlap': 200
    },
    'model': {
        'name': 'gpt-3.5-turbo',
        'temperature': 0,
        'max_tokens': 1000
    }
}
```

## Development Tools

### Code Formatting
```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Linting
```bash
# Run linter
flake8 src/ tests/

# Run type checker
mypy src/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

## Common Issues

### Elasticsearch Connection
If Elasticsearch is not responding:
```bash
# Check if running
docker ps | grep elasticsearch

# Check logs
docker logs elasticsearch

# Restart container
docker-compose restart elasticsearch
```

### Python Dependencies
If you encounter dependency conflicts:
```bash
# Clean environment
pip uninstall -y -r <(pip freeze)

# Reinstall dependencies
pip install -r requirements.txt
```

### Memory Issues
If Elasticsearch fails to start:
```bash
# Increase virtual memory
sudo sysctl -w vm.max_map_count=262144
```

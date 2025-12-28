# Embedding Model Options

## 🎯 Quick Answer

You have **hundreds of models** to choose from! Here are the best options organized by use case:

---

## 📊 Recommended Models by Use Case

### 🚀 For Development & Testing (Fast)

**Best Choice: MiniLM-L6**

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

- ⚡ **Speed:** Very Fast (~5x faster than MPNet)
- 📦 **Size:** ~80MB
- 🎯 **Quality:** Good (80-85% of MPNet quality)
- 💡 **Use When:** Rapid prototyping, testing, limited resources

---

### ⚖️ For Production (Balanced) - **CURRENT DEFAULT**

**Best Choice: MPNet**

```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

- ⚡ **Speed:** Medium
- 📦 **Size:** ~420MB
- 🎯 **Quality:** Excellent (best balance)
- 💡 **Use When:** Production deployments, general purpose

**Alternative: DistilRoBERTa**

```bash
EMBEDDING_MODEL=sentence-transformers/all-distilroberta-v1
EMBEDDING_DIMENSION=768
```

- Similar quality to MPNet
- Slightly faster
- Good alternative

---

### 🏆 For Maximum Quality (Slower)

**Best Choice: RoBERTa Large**

```bash
EMBEDDING_MODEL=sentence-transformers/all-roberta-large-v1
EMBEDDING_DIMENSION=1024
```

- ⚡ **Speed:** Slow (~3x slower than MPNet)
- 📦 **Size:** ~1.3GB
- 🎯 **Quality:** Best possible
- 💡 **Use When:** Quality is critical, resources available

---

### 🌍 For Multilingual Support

**Best Choice: Multilingual MPNet**

```bash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

- 🌐 **Languages:** 50+ languages
- 🎯 **Quality:** Excellent for multilingual
- 💡 **Use When:** Non-English or mixed-language content

**Alternative: Multilingual MiniLM (Faster)**

```bash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384
```

- Faster than MPNet
- Good quality for 50+ languages

---

### 📚 For Long Documents

**Best Choice: LongFormer**

```bash
EMBEDDING_MODEL=sentence-transformers/allenai-longformer-base-4096
EMBEDDING_DIMENSION=768
```

- 📄 **Max Length:** 4096 tokens (vs 512 for others)
- 💡 **Use When:** Very long documents, legal texts, research papers

---

### 🔬 For Scientific/Technical Content

**Best Choice: SciBERT**

```bash
EMBEDDING_MODEL=sentence-transformers/allenai-specter
EMBEDDING_DIMENSION=768
```

- 🔬 **Trained On:** Scientific papers
- 💡 **Use When:** Academic, medical, technical content

---

## 📋 Complete Model Comparison Table

| Model                              | Dimension | Size  | Speed      | Quality    | Best For                 |
| ---------------------------------- | --------- | ----- | ---------- | ---------- | ------------------------ |
| **all-MiniLM-L6-v2**               | 384       | 80MB  | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | Development, Testing     |
| **all-MiniLM-L12-v2**              | 384       | 120MB | ⚡⚡⚡⚡   | ⭐⭐⭐⭐   | Balanced Fast            |
| **all-mpnet-base-v2**              | 768       | 420MB | ⚡⚡⚡     | ⭐⭐⭐⭐⭐ | **Production (Default)** |
| **all-distilroberta-v1**           | 768       | 290MB | ⚡⚡⚡     | ⭐⭐⭐⭐   | Production Alternative   |
| **all-roberta-large-v1**           | 1024      | 1.3GB | ⚡         | ⭐⭐⭐⭐⭐ | Maximum Quality          |
| **paraphrase-multilingual-mpnet**  | 768       | 970MB | ⚡⚡       | ⭐⭐⭐⭐⭐ | Multilingual             |
| **paraphrase-multilingual-MiniLM** | 384       | 420MB | ⚡⚡⚡     | ⭐⭐⭐⭐   | Multilingual Fast        |

---

## 🎨 Specialized Models

### Code Search

```bash
EMBEDDING_MODEL=sentence-transformers/multi-qa-mpnet-base-cos-v1
EMBEDDING_DIMENSION=768
```

- Optimized for question-answering
- Great for FAQ systems

### Semantic Similarity

```bash
EMBEDDING_MODEL=sentence-transformers/paraphrase-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

- Best for finding similar sentences
- Duplicate detection

### Asymmetric Search (Query ≠ Document)

```bash
EMBEDDING_MODEL=sentence-transformers/msmarco-distilbert-base-v4
EMBEDDING_DIMENSION=768
```

- Optimized for short query → long document
- Search engines, Q&A systems

---

## 🔍 How to Choose?

### Decision Tree

```
Start Here
    ↓
Do you need multilingual support?
    ├─ YES → paraphrase-multilingual-mpnet-base-v2
    └─ NO → Continue
         ↓
    What's your priority?
         ├─ Speed → all-MiniLM-L6-v2
         ├─ Balance → all-mpnet-base-v2 (DEFAULT)
         └─ Quality → all-roberta-large-v1
```

### By Resource Constraints

**Limited RAM (<4GB):**

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

**Standard Server (4-8GB):**

```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

**High-End Server (>8GB):**

```bash
EMBEDDING_MODEL=sentence-transformers/all-roberta-large-v1
EMBEDDING_DIMENSION=1024
```

---

## 🧪 How to Test Different Models

### Quick Test Script

```bash
#!/bin/bash

# Test different models and compare results

MODELS=(
    "sentence-transformers/all-MiniLM-L6-v2:384"
    "sentence-transformers/all-mpnet-base-v2:768"
    "sentence-transformers/all-roberta-large-v1:1024"
)

TEST_QUERY="What is a database?"

for model_config in "${MODELS[@]}"; do
    MODEL=$(echo $model_config | cut -d':' -f1)
    DIM=$(echo $model_config | cut -d':' -f2)

    echo "Testing: $MODEL"

    # Update .env
    sed -i '' "s|EMBEDDING_MODEL=.*|EMBEDDING_MODEL=$MODEL|" .env
    sed -i '' "s|EMBEDDING_DIMENSION=.*|EMBEDDING_DIMENSION=$DIM|" .env

    # Restart
    docker-compose restart rag-backend-cli
    sleep 30

    # Reindex
    docker-compose exec rag-backend-cli python main.py index --force-rebuild

    # Test query
    echo "Query: $TEST_QUERY"
    docker-compose exec rag-backend-cli python main.py query -q "$TEST_QUERY"

    echo "---"
done
```

---

## 📖 Find More Models

### Browse HuggingFace

Visit: https://huggingface.co/sentence-transformers

**Popular Collections:**

- General Purpose: https://huggingface.co/sentence-transformers?search=all-
- Multilingual: https://huggingface.co/sentence-transformers?search=multilingual
- Domain-Specific: https://huggingface.co/sentence-transformers?search=domain

### Check Model Details

For any model, visit:

```
https://huggingface.co/sentence-transformers/MODEL-NAME
```

Example:

```
https://huggingface.co/sentence-transformers/all-mpnet-base-v2
```

Look for:

- **Embedding Dimension** (must match EMBEDDING_DIMENSION in .env)
- **Max Sequence Length** (how long texts can be)
- **Performance Metrics** (quality scores)
- **Model Size** (download size)

---

## 🎯 Step-by-Step: Change Model

### Example: Switch to MiniLM for faster development

**Step 1: Edit .env**

```bash
nano .env

# Change these lines:
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

**Step 2: Restart**

```bash
docker-compose restart rag-backend-cli
```

**Step 3: Watch Download**

```bash
docker-compose logs -f rag-backend-cli

# You'll see:
# "Loading local embedding model: sentence-transformers/all-MiniLM-L6-v2"
# "Downloading model from HuggingFace: ..."
# "Local model loaded successfully with dimension: 384"
```

**Step 4: Reindex**

```bash
docker-compose exec rag-backend-cli python main.py index --force-rebuild
```

**Step 5: Test**

```bash
docker-compose exec rag-backend-cli python main.py query -q "What is a database?"
```

---

## ⚠️ Important Notes

### 1. Dimension Must Match

```bash
# ❌ WRONG - Dimension mismatch
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=768  # Wrong! MiniLM is 384

# ✅ CORRECT
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384  # Correct!
```

### 2. Always Reindex After Changing Models

Different models = different embeddings = must reindex

```bash
docker-compose exec rag-backend-cli python main.py index --force-rebuild
```

### 3. Model Download Time

First time using a model:

- Small (MiniLM): ~1-2 minutes
- Medium (MPNet): ~3-5 minutes
- Large (RoBERTa): ~5-10 minutes

Subsequent uses: ~5-10 seconds (loaded from cache)

### 4. Check Model Compatibility

All `sentence-transformers` models work, but verify:

- Model exists on HuggingFace
- Model is for sentence embeddings (not classification, etc.)
- Dimension is documented

---

## 🚀 Quick Reference

### Copy-Paste Configurations

**Fast Development:**

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

**Production (Default):**

```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

**Maximum Quality:**

```bash
EMBEDDING_MODEL=sentence-transformers/all-roberta-large-v1
EMBEDDING_DIMENSION=1024
```

**Multilingual:**

```bash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

---

## 📊 Performance Benchmarks

Based on MTEB (Massive Text Embedding Benchmark):

| Model                | Avg Score | Speed (sentences/sec) |
| -------------------- | --------- | --------------------- |
| all-MiniLM-L6-v2     | 56.3      | ~3000                 |
| all-MiniLM-L12-v2    | 59.8      | ~1500                 |
| all-mpnet-base-v2    | **63.3**  | ~800                  |
| all-distilroberta-v1 | 61.5      | ~900                  |
| all-roberta-large-v1 | 64.6      | ~300                  |

**Higher score = better quality**

---

## ✅ Summary

**You have 3 main options:**

1. **Fast & Small** → `all-MiniLM-L6-v2` (384-dim)
2. **Balanced** → `all-mpnet-base-v2` (768-dim) ← **Default**
3. **Best Quality** → `all-roberta-large-v1` (1024-dim)

**Plus specialized options for:**

- Multilingual content
- Long documents
- Scientific/technical text
- Code search
- And more!

**All models:**

- Download automatically from HuggingFace
- Work with your existing system
- Just edit `.env` and restart!

🎉 **No code changes needed!**

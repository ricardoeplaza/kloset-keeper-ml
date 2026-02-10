# Kloset-Keeper AI Service

This is the machine learning microservice for the **Kloset-Keeper** ecosystem. It provides specialized endpoints for garment image processing, including background removal and vector embedding generation for semantic search.

## Features

* **Background Removal**: Uses `rembg` with the `u2netp` model (CPU optimized).
* **Auto-Crop (Trim)**: Automatically crops transparent borders to focus on the garment.
* **Visual Embeddings**: Generates 512-dimension vectors using the **OpenAI CLIP (ViT-B-32)** model via `sentence-transformers`.

## Tech Stack

* **Framework**: FastAPI
* **Imaging**: Pillow (PIL)
* **ML Libraries**: Rembg, Sentence-Transformers (PyTorch based)
* **Deployment**: Ideal for Docker or native Python 3.1x.

## API Endpoints

### 1. Remove Background

`POST /remove-background`

* **Input**: Multipart Image File.
* **Output**: Transparent PNG file (cropped to garment bounds).

### 2. Generate Embedding

`POST /generate-embedding`

* **Input**: Multipart Image File (Recommended: 300px WebP).
* **Output**: JSON containing the 512-float vector.

## Self-Hosted Setup

```bash
# Clone the repository
git clone https://github.com/ricardoeplaza/kloset-keeper-ml.git

# Create virtual environment
python -m venv venv

# Activate venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn rembg sentence-transformers pillow python-multipart

# Run the service
uvicorn main:app --host 0.0.0.0 --port 8000

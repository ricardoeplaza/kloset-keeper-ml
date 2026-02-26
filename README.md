# Kloset-Keeper AI Service

This is the machine learning microservice for the **Kloset-Keeper** ecosystem. It provides specialized endpoints for garment image processing, including background removal and multimodal vector embedding generation (CLIP) optimized for CPU.

## 🚀 Key Update: OpenVINO Integration

We have migrated the inference backend to **OpenVINO (Open Visual Inference and Neural Network Optimization)**.

### What is OpenVINO?

Developed by Intel, OpenVINO is a toolkit designed to optimize and deploy AI inference on various hardware. In this project, it specifically targets the **CPU** to achieve near-GPU performance without needing a dedicated graphics card.

### Improvements over Default Backend (PyTorch/CPU)

* **Latency Reduction**: Significant speedup in generating embeddings (CLIP) by utilizing advanced instruction sets (AVX-512, VNNI).
* **Model Quantization**: Efficient handling of weights which reduces memory footprint on your server.
* **Throughput**: Better handling of concurrent requests, ideal for the microservice architecture of Kloset-Keeper.

---

## Features

* **Background Removal**: Uses `rembg` with the `u2netp` model.
* **Auto-Crop (Trim)**: Automatically cleans "ghost" pixels and crops transparent borders to focus strictly on the garment.
* **Multimodal Embeddings**: Generates 512-dimension vectors using **OpenAI CLIP (ViT-B-32)** model via `sentence-transformers`.

## Tech Stack

* **Framework**: FastAPI
* **Inference Engine**: OpenVINO
* **ML Libraries**: Rembg, Sentence-Transformers, NumPy
* **Imaging**: Pillow (PIL)

---

## API Endpoints

### 0. Health Check

`GET /`
Returns the server status, version, and a list of available endpoints.

### 1. Remove Background

`POST /remove-background`

* **Input**: Multipart Image File.
* **Output**: Transparent PNG file (alpha-matting and auto-crop applied).

### 2. Text Embedding

`POST /embeddings/text`

* **Input**: `text` (Form data).
* **Output**: JSON with the 512-float vector.

### 3. Image/Multimodal Embedding

`POST /embeddings/image`

* **Input**: `file` (Image) and optional `description` (Text).
* **Logic**: If both are provided, it performs a weighted average (70% image / 30% text) to refine the visual search.

---

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

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the service
uvicorn main:app --host 0.0.0.0 --port 8000

```

from fastapi import FastAPI, UploadFile, File, Response
from rembg import remove, new_session
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

app = FastAPI(title="Garment AI Processor")

# Optimized session for CPU (u2netp is lightweight and fast for Debian environments)
bg_session = new_session("u2netp")

# CLIP Model: Industry standard for visual embeddings
# Note: Internally resizes to 224x224
embed_model = SentenceTransformer('clip-ViT-B-32', device='cpu')

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """
    Receives an original image (multipart) and returns a transparent PNG
    with auto-cropping (trim) applied.
    """
    input_data = await file.read()
    
    # Background removal process
    output_png = remove(
        input_data, 
        session=bg_session, 
        alpha_matting_foreground_threshold=245,
        alpha_matting_background_threshold=10
    )
    
    # TRIM (Auto-crop) logic using Pillow
    # This removes extra transparent pixels around the garment
    img = Image.open(io.BytesIO(output_png))
    bbox = img.getbbox() # Finds the bounding box of non-transparent data
    if bbox:
        img = img.crop(bbox)
    
    # Convert back to bytes for the response
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/generate-embedding")
async def generate_embedding(file: UploadFile = File(...)):
    """
    Receives a 300px WebP thumbnail and returns a 512-dimension vector
    """
    input_data = await file.read()
    
    # Open WebP and ensure RGB conversion for CLIP compatibility
    image = Image.open(io.BytesIO(input_data)).convert("RGB")
    
    # Generate 512-dimension embedding vector
    embedding = embed_model.encode(image)
    
    return {
        "embedding": embedding.tolist(),
        "dimensions": 512,
        "model": "clip-ViT-B-32"
    }
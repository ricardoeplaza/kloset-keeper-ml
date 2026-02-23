from fastapi import FastAPI, UploadFile, File, Form, Response
from rembg import remove, new_session
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np

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
    
    # 1. Background removal process
    output_png = remove(
        input_data, 
        session=bg_session, 
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=15,
        alpha_matting_erode_size=10 # Small erosion helps with dark halos
    )
    
    # 2. Open image with Pillow for post-processing
    img = Image.open(io.BytesIO(output_png))

    # 3. Clean alpha channel (Remove faint shadows/ghost pixels)
    # This ensures getbbox() doesn't include unwanted residues
    r, g, b, a = img.split()
    # Threshold: any pixel with alpha < 30 becomes 0 (fully transparent)
    a = a.point(lambda p: p if p > 30 else 0)
    img.putalpha(a)

    # # 4. Brightness & Contrast Enhancement
    # # Increase brightness slightly (1.1 = +10%) and contrast (1.1 = +10%)
    # # This compensates for the dimming effect in the edges
    # enhancer_bright = ImageEnhance.Brightness(img)
    # img = enhancer_bright.enhance(1.05) 
    
    # enhancer_cont = ImageEnhance.Contrast(img)
    # img = enhancer_cont.enhance(1.1)

    # 5. TRIM (Auto-crop) logic
    # Now bbox will be much more accurate after cleaning the alpha channel
    bbox = img.getbbox() 
    if bbox:
        img = img.crop(bbox)
    
    # 6. Convert back to bytes for the response
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/embeddings/text")
async def get_text_embedding(text: str = Form(...)):
    """Generates an embedding for a single text string (Categories, Search)."""
    embedding = embed_model.encode(text)
    return {
        "embedding": normalize_vector(embedding).tolist(),
        "model": "clip-ViT-B-32",
        "dimensions": 512
    }

@app.post("/embeddings/image")
async def get_multimodal_embedding(file: UploadFile = File(...), description: str = Form(None)):
    """
    Generates a multimodal embedding. If description is provided, it fuses image and text.
    """
    # 1. Process Image Embedding
    input_data = await file.read()
    image = Image.open(io.BytesIO(input_data)).convert("RGB")
    img_embedding = embed_model.encode(image)
    
    final_embedding = img_embedding

    # 2. Optional Process Text Embedding
    if description:
        # Generate text embedding using the same CLIP model
        text_embedding = embed_model.encode(description)
        
        # Simple Weighted Average (70% image, 30% text as an example)
        # You can adjust these weights based on your preference
        w_img, w_txt = 0.7, 0.3
        
        final_embedding = (img_embedding * w_img) + (text_embedding * w_txt)

    return {
        "embedding": normalize_vector(final_embedding).tolist(),
        "dimensions": 512,
        "model": "clip-ViT-B-32",
        "refined_by_text": bool(description)
    }
    
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
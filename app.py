"""Project 2 API: serves your week 6 image model and Week 7 text evidence."""
import io
import os
import re
import time
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms

MODEL_PATH = os.getenv('MODEL_PATH', 'week6_model.pth')
POSITIVE = os.getenv('POSITIVE', 'PNEUMONIA')
THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))
API_KEY = os.getenv('API_KEY')
MAX_BYTES = 5 * 1024 * 1024  # 5 MB limit

URGENT = ['fever', 'breath', 'cough', 'chest pain', 'spreading', 'crack']

PATTERNS = {
    'cnic': r'\b\d{5}-\d{7}-\d\b',
    'mobile': r'(?:\+92[-]?|0)3\d{2}[-]?\d{7}\b',
    'amount': r'(?:PKR|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)',
    'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'
}

state = {}

@asynccontextmanager
async def lifespan(app):
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        classes = ckpt.get('classes', ['NORMAL', 'PNEUMONIA'])
        img_size = ckpt.get('img_size', 224)
        grayscale = ckpt.get('grayscale', False)
        backbone_name = ckpt.get('backbone', 'resnet18')
    else:
        state_dict = ckpt
        classes = ['NORMAL', 'PNEUMONIA']
        img_size = 224
        grayscale = False
        backbone_name = 'resnet18'

    out_features = len(classes)

    # Build model backbone & classification head
    if 'efficientnet' in backbone_name.lower():
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, out_features)
        head_keys = ['classifier.1.weight', 'classifier.1.bias']
    else:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, out_features)
        head_keys = ['fc.weight', 'fc.bias']

    # Drop mismatched head weights if checkpoint shape differs (e.g. 1000 vs 2 classes)
    for k in head_keys:
        if k in state_dict and state_dict[k].shape[0] != out_features:
            del state_dict[k]

    m.load_state_dict(state_dict, strict=False)
    m.eval()

    state['ck'] = {'classes': classes, 'img_size': img_size, 'grayscale': grayscale}
    state['model'] = m
    state['tf'] = transforms.Compose(
        ([transforms.Grayscale(3)] if grayscale else []) +
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    yield
    state.clear()

app = FastAPI(title='Project 2 Vision + Text API', version='1.0', lifespan=lifespan)

class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    latency_ms: float

class CaseRecord(BaseModel):
    prediction: Prediction
    note_masked: str
    fields: dict[str, list[str]]
    urgent_terms: list[str]
    decision: str
    reasons: list[str]
    disclaimer: str

def check_key(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, 'Invalid or missing API key')

async def read_image(file: UploadFile):
    if file.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(415, 'Upload a JPEG or PNG image')
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image larger than 5 MB')
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, 'File is not a readable image')

def classify(img):
    ck, to = state['ck'], time.perf_counter()
    x = state['tf'](img.convert('L' if ck['grayscale'] else 'RGB'))
    with torch.inference_mode():
        p = F.softmax(state['model'](x.unsqueeze(0)), dim=1)[0].tolist()
    probs = {c: round(v, 4) for c, v in zip(ck['classes'], p)}
    return Prediction(label=max(probs, key=probs.get), probabilities=probs,
                      latency_ms=round((time.perf_counter() - to) * 1000, 1))

def mask_pii(text):
    for key, tag in [('cnic', '[CNIC]'), ('mobile', '[PHONE]')]:
        text = re.sub(PATTERNS[key], tag, text)
    return text

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'model' in state,
            'classes': state['ck']['classes'], 'positive': POSITIVE}

@app.post('/predict', response_model=Prediction)
async def predict(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    img = await read_image(file)
    return await run_in_threadpool(classify, img)

@app.post('/analyze', response_model=CaseRecord)
async def analyze(
    file: UploadFile = File(...), 
    note: str = Form(...), 
    x_api_key: str | None = Header(None)
):
    check_key(x_api_key)
    if len(note) > 2000:
        raise HTTPException(422, 'Note longer than 2,000 characters')
    pred = await run_in_threadpool(classify, await read_image(file))
    safe = mask_pii(note)
    fields = {k: re.findall(p, safe, flags=re.IGNORECASE) for k, p in PATTERNS.items()}
    urgent = [w for w in URGENT if w in safe.lower()]
    reasons = []
    if pred.probabilities.get(POSITIVE, 0) >= THRESHOLD:
        reasons.append(f'image: P({POSITIVE}) = {pred.probabilities[POSITIVE]:.2f}')
    if urgent:
        reasons.append('note mentions: ' + ', '.join(urgent))
    return CaseRecord(
        prediction=pred, note_masked=safe,
        fields={k: v for k, v in fields.items() if v},
        urgent_terms=urgent, decision='REFER' if reasons else 'ROUTINE',
        reasons=reasons,
        disclaimer='Decision support only; a qualified human decides.'
    )    
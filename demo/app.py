"""Chamber Hot Dog Demo — Is it a Hot Dog?
Open on your phone, upload or snap a photo, get instant prediction.
Inspired by Silicon Valley S4E4.
"""
import os, io, base64, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

CLASSES = ["not_hot_dog", "hot_dog"]
MODEL_PATH = os.environ.get("MODEL_PATH", "../model/best_model.pth")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/vit-base-patch16-224")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model at startup
print(f"Loading model from {MODEL_PATH}...")
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image):
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        logits = model(tensor).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
    elapsed = (time.time() - start) * 1000
    return {
        "prediction": CLASSES[pred_idx],
        "confidence": f"{probs[pred_idx].item() * 100:.1f}%",
        "probabilities": {
            "hot_dog": round(probs[1].item() * 100, 1),
            "not_hot_dog": round(probs[0].item() * 100, 1),
        },
        "inference_ms": round(elapsed, 1),
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Hot Dog / Not Hot Dog</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0e17; color: #e0e0e0; min-height: 100vh; }
.container { max-width: 500px; margin: 0 auto; padding: 20px; }
.header { text-align: center; padding: 30px 0 20px; }
.header h1 { font-size: 28px; color: #fff; margin-bottom: 4px; }
.header .sub { font-size: 14px; color: #64748b; }
.header .powered { font-size: 11px; color: #0ea5e9; margin-top: 8px; }
.header .ref { font-size: 11px; color: #475569; margin-top: 4px; font-style: italic; }
.upload-area { border: 2px dashed #1e293b; border-radius: 16px; padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.2s; margin: 20px 0; background: #0f172a; }
.upload-area:hover, .upload-area.dragover { border-color: #f59e0b; background: #1a150a; }
.upload-area .icon { font-size: 64px; margin-bottom: 12px; }
.upload-area p { color: #94a3b8; font-size: 14px; }
.upload-area .or { color: #475569; font-size: 12px; margin: 8px 0; }
.btn-sample { display: inline-block; background: #334155; color: #e0e0e0; border: none; margin-top: 12px; font-size: 13px; padding: 10px 20px; border-radius: 10px; cursor: pointer; transition: all 0.2s; }
.btn-sample:hover { background: #475569; }
.preview { margin: 20px 0; text-align: center; }
.preview img { max-width: 100%; max-height: 300px; border-radius: 12px; border: 1px solid #1e293b; }
.result { margin: 20px 0; border-radius: 16px; padding: 24px; text-align: center; animation: fadeIn 0.3s; }
.result.hot_dog { background: linear-gradient(135deg, #78350f, #0f172a); border: 1px solid #f59e0b; }
.result.not_hot_dog { background: linear-gradient(135deg, #7f1d1d, #0f172a); border: 1px solid #ef4444; }
.result .emoji { font-size: 64px; margin-bottom: 8px; }
.result .label { font-size: 32px; font-weight: 700; margin-bottom: 4px; }
.result.hot_dog .label { color: #fbbf24; }
.result.not_hot_dog .label { color: #f87171; }
.result .conf { font-size: 16px; color: #94a3b8; }
.result .time { font-size: 12px; color: #475569; margin-top: 8px; }
.probs { display: flex; gap: 12px; justify-content: center; margin-top: 16px; }
.prob-bar { flex: 1; max-width: 180px; }
.prob-bar .bar-label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
.prob-bar .bar-bg { background: #1e293b; border-radius: 6px; height: 8px; overflow: hidden; }
.prob-bar .bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s; }
.prob-bar .bar-fill.hotdog { background: #f59e0b; }
.prob-bar .bar-fill.nothotdog { background: #ef4444; }
.prob-bar .bar-value { font-size: 12px; color: #cbd5e1; margin-top: 2px; text-align: right; }
.spinner { display: none; text-align: center; padding: 30px; }
.spinner.active { display: block; }
.spinner .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #f59e0b; margin: 0 4px; animation: bounce 1.4s infinite; }
.spinner .dot:nth-child(2) { animation-delay: 0.2s; }
.spinner .dot:nth-child(3) { animation-delay: 0.4s; }
.stats { margin-top: 30px; padding: 16px; background: #0f172a; border-radius: 12px; border: 1px solid #1e293b; }
.stats h3 { font-size: 13px; color: #64748b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.stats .row { display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; }
.stats .row .val { color: #f59e0b; font-weight: 600; }
input[type="file"] { display: none; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-10px); } }
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🌭 Hot Dog / Not Hot Dog</h1>
        <div class="sub">AI-Powered Food Classification</div>
        <div class="powered">Powered by Chamber GPU Cloud</div>
        <div class="ref">Inspired by Silicon Valley S4E4</div>
    </div>

    <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
        <div class="icon">📸</div>
        <p>Tap to upload a food photo</p>
        <div class="or">or drag & drop</div>
        <input type="file" id="fileInput" accept="image/*" capture="environment">
    </div>

    <div style="text-align: center;">
        <button class="btn-sample" onclick="loadSample('hotdog')">Try Sample: 🌭 Hot Dog</button>
        <button class="btn-sample" onclick="loadSample('not_hotdog')">Try Sample: ❌ Not Hot Dog</button>
    </div>

    <div class="preview" id="preview" style="display:none;">
        <img id="previewImg" src="">
    </div>

    <div class="spinner" id="spinner">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        <p style="margin-top: 12px; color: #64748b; font-size: 13px;">Analyzing food...</p>
    </div>

    <div id="result"></div>

    <div class="stats">
        <h3>Model Info</h3>
        <div class="row"><span>Architecture</span><span class="val">ViT-B/16 (HuggingFace)</span></div>
        <div class="row"><span>Training</span><span class="val">Pretrained + Fine-tuned</span></div>
        <div class="row"><span>Dataset</span><span class="val">Food-101 (hot_dog subset)</span></div>
        <div class="row"><span>GPU</span><span class="val">Tesla T4 (Chamber)</span></div>
        <div class="row"><span>W&B Project</span><span class="val">chamber-hotdog-demo</span></div>
    </div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

['dragenter', 'dragover'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(e => dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('dragover'); }));
dropZone.addEventListener('drop', e => { if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById('preview').style.display = 'block';
        document.getElementById('previewImg').src = e.target.result;
        classify(e.target.result);
    };
    reader.readAsDataURL(file);
}

function loadSample(type) {
    document.getElementById('spinner').classList.add('active');
    document.getElementById('result').innerHTML = '';
    fetch('/sample/' + type)
        .then(r => r.json())
        .then(data => {
            document.getElementById('preview').style.display = 'block';
            document.getElementById('previewImg').src = 'data:image/jpeg;base64,' + data.image;
            showResult(data.result);
        });
}

function classify(dataUrl) {
    document.getElementById('spinner').classList.add('active');
    document.getElementById('result').innerHTML = '';
    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image: dataUrl})
    })
    .then(r => r.json())
    .then(data => showResult(data))
    .catch(err => {
        document.getElementById('spinner').classList.remove('active');
        document.getElementById('result').innerHTML = '<p style="color:#ef4444;">Error: ' + err + '</p>';
    });
}

function showResult(data) {
    document.getElementById('spinner').classList.remove('active');
    const cls = data.prediction;
    const emoji = cls === 'hot_dog' ? '🌭' : '❌';
    const displayLabel = cls === 'hot_dog' ? 'HOT DOG!' : 'NOT HOT DOG';
    const probs = data.probabilities;
    document.getElementById('result').innerHTML = `
        <div class="result ${cls}">
            <div class="emoji">${emoji}</div>
            <div class="label">${displayLabel}</div>
            <div class="conf">${data.confidence} confidence</div>
            <div class="probs">
                <div class="prob-bar">
                    <div class="bar-label">🌭 Hot Dog</div>
                    <div class="bar-bg"><div class="bar-fill hotdog" style="width:${probs.hot_dog}%"></div></div>
                    <div class="bar-value">${probs.hot_dog}%</div>
                </div>
                <div class="prob-bar">
                    <div class="bar-label">❌ Not Hot Dog</div>
                    <div class="bar-bg"><div class="bar-fill nothotdog" style="width:${probs.not_hot_dog}%"></div></div>
                    <div class="bar-value">${probs.not_hot_dog}%</div>
                </div>
            </div>
            <div class="time">Inference: ${data.inference_ms}ms</div>
        </div>
    `;
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    result = predict(img)
    return jsonify(result)


@app.route("/sample/<sample_type>")
def sample(sample_type):
    from datasets import load_dataset
    ds = load_dataset("ethz/food101", split="validation", trust_remote_code=True)
    label_names = ds.features["label"].names
    hotdog_idx = label_names.index("hot_dog")
    target = hotdog_idx if sample_type == "hotdog" else None

    for item in ds:
        if sample_type == "hotdog" and item["label"] == hotdog_idx:
            img = item["image"].convert("RGB")
        elif sample_type == "not_hotdog" and item["label"] != hotdog_idx:
            img = item["image"].convert("RGB")
        else:
            continue
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        result = predict(img)
        return jsonify({"image": b64, "result": result})
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    print("\n🌭 Chamber Hot Dog Demo")
    print("   Open on your phone: http://<your-ip>:5051")
    print("   Is it a hot dog?\n")
    app.run(host="0.0.0.0", port=5051, debug=False)

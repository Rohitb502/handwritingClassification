"""Handwriting writer-verification website.

Pipeline per uploaded image:
    grayscale -> Otsu binarization -> OCR (Tesseract) line detection -> line crops
Then 10 crop pairs (image 1 vs image 2) are embedded and compared, and the
scores are aggregated (mean score + vote count).

Run:  python app.py        then open http://127.0.0.1:5000
Env:  MODEL_PATH     path to your .pt checkpoint (default: best_verifier.pt)
      THRESHOLD      override the cosine threshold stored in the checkpoint
      TESSERACT_CMD  full path to tesseract binary (Windows, if not on PATH)

Install: pip install flask torch torchvision pillow opencv-python pytesseract numpy
         + the Tesseract binary (sudo apt install tesseract-ocr, or the Windows installer)
"""
import io
import os
import random
from itertools import product

import cv2
import numpy as np
import pytesseract
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

MODEL_PATH = os.environ.get("MODEL_PATH", "best_verifier.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_COMPARISONS = 10

if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


# ---------------- model (same as training notebook) ----------------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        idn = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idn = self.downsample(x)
        return F.relu(out + idn, inplace=True)


class ResNetBackbone(nn.Module):
    def __init__(self, layers, in_ch=1, embed_dim=512):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make(64, layers[0], 1)
        self.layer2 = self._make(128, layers[1], 2)
        self.layer3 = self._make(256, layers[2], 2)
        self.layer4 = self._make(512, layers[3], 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embed_dim)

    def _make(self, planes, n, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        blocks = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        blocks += [BasicBlock(planes, planes) for _ in range(1, n)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.gap(x).flatten(1))


class _ArcStub(nn.Module):
    """Holds the ArcFace class weights so the checkpoint loads. Unused at inference."""
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))


class HWVerifier(nn.Module):
    def __init__(self, layers, embed_dim, n_classes):
        super().__init__()
        self.backbone = ResNetBackbone(layers, 1, embed_dim)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.arc = _ArcStub(embed_dim, n_classes)

    def embed(self, x):
        return F.normalize(self.emb_bn(self.backbone(x)))


# ---------------- load checkpoint ----------------
def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    is_dict = isinstance(ckpt, dict)
    state = ckpt["model"] if is_dict and "model" in ckpt else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    cfg = ckpt.get("config", {}) if is_dict else {}

    model = HWVerifier(cfg.get("LAYERS", [3, 4, 6, 3]),
                       cfg.get("EMBED_DIM", 512),
                       state["arc.weight"].shape[0])
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    threshold = ckpt.get("best_thr", ckpt.get("eer_thr", 0.5)) if is_dict else 0.5
    if os.environ.get("THRESHOLD"):
        threshold = float(os.environ["THRESHOLD"])
    return (model, float(threshold),
            cfg.get("IMG_H", 128), cfg.get("IMG_W", 512), cfg.get("PAD_VALUE", 255))


MODEL, THRESHOLD, IMG_H, IMG_W, PAD_VALUE = load_model(MODEL_PATH)
to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])


# ---------------- preprocessing ----------------
def fit_image(img):
    """Same geometry as training: grayscale, height 128, pad/squash width to 512."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L")
    w, h = img.size
    new_w = max(1, int(round(w * IMG_H / h)))
    img = img.resize((new_w, IMG_H), Image.BILINEAR)
    if new_w > IMG_W:
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    canvas = Image.new("L", (IMG_W, IMG_H), color=PAD_VALUE)
    canvas.paste(img, (0, 0))
    return canvas


def otsu_binarize(img):
    """PIL image -> grayscale -> Otsu. Returns binary uint8 array (ink=0, paper=255)."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    gray = np.array(img.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def ocr_line_crops(binary, pad=6, min_w=40, min_h=12, min_conf=30):
    """OCR the binarized page, group words into lines, return line crops (PIL, mode L).

    OCR is used only to locate text lines. If Tesseract is missing, errors out, or
    finds nothing, the whole binarized image is used as a single crop.
    """
    whole = [Image.fromarray(binary)]
    try:
        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError):
        return whole

    lines = {}
    for i, txt in enumerate(data["text"]):
        try:
            conf = float(data["conf"][i])
        except ValueError:
            continue
        if not txt.strip() or conf < min_conf:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = (int(data[k][i]) for k in ("left", "top", "width", "height"))
        x0, y0, x1, y1 = lines.get(key, (10 ** 9, 10 ** 9, 0, 0))
        lines[key] = (min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h))

    H, W = binary.shape
    crops = []
    for x0, y0, x1, y1 in lines.values():
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
        if x1 - x0 >= min_w and y1 - y0 >= min_h:
            crops.append(Image.fromarray(binary[y0:y1, x0:x1]))
    return crops or whole


def make_pairs(na, nb, n=N_COMPARISONS):
    """Exactly n (i, j) crop pairs; deterministic. Reuses pairs if there are too few crops."""
    allp = list(product(range(na), range(nb)))
    rng = random.Random(0)
    rng.shuffle(allp)
    return [allp[k % len(allp)] for k in range(n)]


def read_upload(field):
    f = request.files.get(field)
    if f is None or f.filename == "":
        raise ValueError("Add both images before comparing.")
    try:
        return Image.open(io.BytesIO(f.read()))
    except (UnidentifiedImageError, OSError):
        raise ValueError(f"“{f.filename}” isn’t a readable image.")


# ---------------- web ----------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", threshold=round(THRESHOLD, 3))


@app.post("/compare")
def compare():
    try:
        a, b = read_upload("image1"), read_upload("image2")
    except ValueError as e:
        return jsonify(error=str(e)), 400

    thr = request.form.get("threshold", type=float)
    thr = THRESHOLD if thr is None else thr

    # grayscale -> Otsu -> OCR line crops
    crops_a = ocr_line_crops(otsu_binarize(a))
    crops_b = ocr_line_crops(otsu_binarize(b))

    # embed every crop once, then reuse for all 10 comparisons
    x = torch.stack([to_tensor(fit_image(c)) for c in crops_a + crops_b]).to(DEVICE)
    with torch.no_grad():
        e = MODEL.embed(x).cpu()
    ea, eb = e[:len(crops_a)], e[len(crops_a):]

    pairs = make_pairs(len(crops_a), len(crops_b))
    scores = [float((ea[i] * eb[j]).sum()) for i, j in pairs]
    mean_score = float(np.mean(scores))
    votes = int(sum(s >= thr for s in scores))

    return jsonify(
        score=mean_score,
        scores=scores,
        votes_same=votes,
        votes_total=len(scores),
        threshold=thr,
        same_writer=bool(mean_score >= thr),  # or: votes > len(scores) / 2
        lines_found=[len(crops_a), len(crops_b)],
    )


if __name__ == "__main__":
    print(f"Model: {MODEL_PATH} | device: {DEVICE} | threshold: {THRESHOLD:.3f}")
    app.run(host="127.0.0.1", port=5000, debug=False)
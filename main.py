# ══════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS & ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════
import base64
import csv
import io
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"

import cv2
import easyocr
import google.generativeai as genai
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from inference import get_model
from PIL import Image

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "vnh-detection-binh/1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CROP_PAD_PX = 20


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: API CLIENT FACTORIES
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(["en"], gpu=False)


def get_groq_client():
    key = st.session_state.get("ui_groq_key", "").strip()
    if not key:
        raise ValueError("missing_groq_key")
    return Groq(api_key=key)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: PAGE CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DocStruct — Turn Unstructured Docs Into Structure",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }

:root {
    --bg:        #2b2d33;
    --surface:   #33363e;
    --surface-2: #3a3d47;
    --border:    #464954;
    --accent:    #3b82f6;
    --green:     #10b981;
    --red:       #ef4444;
    --amber:     #f59e0b;
    --text:      #e2e8f0;
    --muted:     #94a3b8;
}

.stApp, .stApp > div { background: var(--bg) !important; }
.block-container { padding-top: 1rem !important; max-width: 100% !important; }

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    overflow-x: hidden !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1rem 0.8rem !important;
    overflow-x: hidden !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] > button,
[data-testid="collapsedControl"] > button,
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[kind="headerNoPadding"] {
    visibility: visible !important;
    opacity: 1 !important;
    display: flex !important;
    background: #3a3d47 !important;
    border: 1px solid #464954 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg,
button[aria-label="Close sidebar"] svg,
button[aria-label="Open sidebar"] svg {
    fill: #e2e8f0 !important;
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:first-child {
    background: var(--border) !important; height: 4px !important; border-radius: 2px !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="progressbar"] {
    background-color: var(--accent) !important; border-radius: 2px !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid white !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.35) !important;
    width: 16px !important; height: 16px !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSlider p { color: var(--text) !important; font-size: 12px !important; }
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: var(--accent) !important; font-size: 11px !important; }
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] { color: var(--muted) !important; font-size: 10px !important; }
section[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 6px 0 !important; }
section[data-testid="stSidebar"] .stSlider { max-width: 100% !important; padding: 0 !important; }
section[data-testid="stSidebar"] .stSlider > div { max-width: 100% !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--muted) !important;
    font-size: 13px !important; padding: 11px 18px !important; border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

.ocr-block {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
    padding: 16px 20px;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    font-size: 14px; line-height: 1.8; direction: ltr; text-align: left;
    color: var(--text); min-height: 80px; white-space: pre-wrap;
}
.ocr-error { color: var(--amber); font-family: monospace; font-size: 12px; }

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 0; overflow: hidden; margin-bottom: 12px;
}
.field-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.field-table tr { border-bottom: 1px solid var(--border); }
.field-table tr:last-child { border-bottom: none; }
.field-table td { padding: 9px 14px; vertical-align: top; }
.field-key  { color: var(--muted); font-weight: 500; width: 38%; white-space: nowrap; }
.field-val  { color: var(--text); }
.field-empty { color: #334155; font-style: italic; }

.stat-row { display: flex; gap: 8px; }
.stat-pill {
    flex: 1; background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 4px; text-align: center;
}
.stat-num { font-size: 18px; font-weight: 700; display: block; line-height: 1.2; }
.stat-lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }

.steps { display: flex; margin-bottom: 14px; }
.step {
    flex: 1; padding: 7px 8px; background: var(--surface); border: 1px solid var(--border);
    font-size: 11px; color: var(--muted); display: flex; align-items: center;
    gap: 6px; margin-right: -1px; white-space: nowrap;
}
.step:first-child { border-radius: 6px 0 0 6px; }
.step:last-child  { border-radius: 0 6px 6px 0; margin-right: 0; }
.step.done  { color:#10b981; background:rgba(16,185,129,.08); border-color:rgba(16,185,129,.2); }
.step.active{ color:#3b82f6; background:rgba(59,130,246,.08); border-color:rgba(59,130,246,.25); }
.step-num {
    width: 15px; height: 15px; border-radius: 50%; border: 1.5px solid currentColor;
    display: flex; align-items: center; justify-content: center;
    font-size: 9px; font-weight: 700; flex-shrink: 0;
}

.sec-label {
    font-size: 9.5px; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 8px; display: block;
}

[data-testid="stImage"] > img {
    max-height: 380px !important; width: auto !important; max-width: 100% !important;
    object-fit: contain !important; display: block; border-radius: 4px; border: 1px solid var(--border);
}

[data-testid="column"] { padding: 0 6px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

[data-testid="stFileUploader"] { margin-bottom: 8px !important; }
[data-testid="stFileUploader"] img { display: none !important; }

.stAlert { background: var(--surface-2) !important; border-color: var(--border) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: CONSTANTS & FIELD DEFINITIONS
# ══════════════════════════════════════════════════════════════════════
ALL_FIELD_NAMES = [
    "Complaint / Ticket ID",
    "Date Submitted",
    "Reported By",
    "Department / Team",
    "Affected System / Product",
    "Software Version",
    "Environment",
    "Issue Type",
    "Severity / Priority",
    "Issue Summary",
    "Steps to Reproduce",
    "Expected Behaviour",
    "Actual Behaviour",
    "Error Message / Code",
    "Attachments Mentioned",
    "Assigned To",
    "Resolution Status",
    "Resolution Summary",
    "Date Resolved",
]


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: DETECTION & IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return get_model(model_id=ROBOFLOW_MODEL_ID, api_key=ROBOFLOW_API_KEY)


def run_inference(image_path):
    return load_model().infer(image_path)[0].predictions


def iou(a, b):
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def postprocess(predictions, img_w, img_h, conf_thresh, iou_thresh):
    boxes = []
    for det in predictions:
        if det.confidence < conf_thresh:
            continue
        x, y, w, h = det.x, det.y, det.width, det.height
        x1 = max(0, int(x - w / 2) - CROP_PAD_PX)
        y1 = max(0, int(y - h / 2) - CROP_PAD_PX)
        x2 = min(img_w, int(x + w / 2) + CROP_PAD_PX)
        y2 = min(img_h, int(y + h / 2) + CROP_PAD_PX)
        boxes.append({"box": (x1, y1, x2, y2), "confidence": det.confidence})
    boxes.sort(key=lambda b: b["confidence"], reverse=True)
    keep = []
    for b in boxes:
        if all(iou(b["box"], k["box"]) < iou_thresh for k in keep):
            keep.append(b)
    keep.sort(key=lambda b: b["box"][1])
    return keep


def draw_boxes(img, detected):
    vis = img.copy()
    for e in detected:
        cv2.rectangle(
            vis, (e["box"][0], e["box"][1]), (e["box"][2], e["box"][3]), (0, 255, 0), 2
        )
    return vis


def crop_regions(img, regions):
    crops = []
    for r in regions:
        x1, y1, x2, y2 = r["box"]
        c = img[y1:y2, x1:x2]
        if c.size > 0:
            crops.append(c)
    return crops


def resize_for_display(img_bgr, max_long_edge=640):
    h, w = img_bgr.shape[:2]
    long = max(h, w)
    if long <= max_long_edge:
        return img_bgr
    s = max_long_edge / long
    return cv2.resize(
        img_bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
    )


def to_display_bytes(img_bgr, quality=85):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: IMAGE PREPROCESSING (DESKEW & ENHANCE)
# ══════════════════════════════════════════════════════════════════════
def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def camscanner_enhance(img):
    img = deskew(img)
    h, w = img.shape[:2]
    scale = 4 if h < 64 else 3 if h < 128 else 2 if h < 256 else 1
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(
        gray, h=8, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)
    padded = cv2.copyMakeBorder(
        sharpened, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


def _color_crop_to_b64(crop):
    img = deskew(crop)
    h, w = img.shape[:2]
    scale = 4 if h < 64 else 3 if h < 128 else 2 if h < 256 else 1
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(
        img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _is_blank_crop(crop, threshold=245):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    return float(np.mean(gray)) > threshold


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: OCR (EASYOCR DRAFT + LLAMA-4 SCOUT VERIFICATION)
# ══════════════════════════════════════════════════════════════════════
def _easyocr_single_crop(crop, reader):
    if _is_blank_crop(crop):
        return ""
    enhanced = camscanner_enhance(crop)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    try:
        results = reader.readtext(
            gray, detail=0, paragraph=True, contrast_ths=0.05, adjust_contrast=0.7
        )
        return " ".join(results).strip()
    except Exception:
        return ""


def _ocr_single_crop(args):
    idx, crop, client, easyocr_text = args
    img_b64 = _color_crop_to_b64(crop)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict verbatim transcriber for IT complaint documents. "
                            "Your ONLY job is to read the image and output EXACTLY the text visible in it — "
                            "character by character, word by word — with NO rephrasing, NO paraphrasing, "
                            "NO corrections of grammar or spelling, NO reordering, and NO additions. "
                            "Copy every word, number, date, code, error message, and punctuation mark "
                            "EXACTLY as it appears in the image. Do NOT fix typos. Do NOT translate. "
                            "Do NOT add labels, commentary, or explanations. "
                            "If nothing is readable, return absolutely nothing."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"An initial OCR engine produced the following rough draft for this line:\n"
                                    f"---\n{easyocr_text}\n---\n"
                                    f"Use the draft ONLY as a reading aid to help you decipher the image. "
                                    f"Do NOT copy the draft. Do NOT rephrase anything. "
                                    f"Look at the image and transcribe VERBATIM — every word, number, and symbol "
                                    f"exactly as written, in the exact same order. Output the raw transcription only."
                                ),
                            },
                        ],
                    },
                ],
                max_tokens=512,
            )
            text = response.choices[0].message.content.strip()
            return idx, " ".join(text.splitlines())
        except Exception as e:
            err = str(e)
            if "rate_limit_exceeded" in err or "429" in err:
                wait_match = re.search(r"try again in (\d+(?:\.\d+)?)s", err)
                wait = float(wait_match.group(1)) + 0.5 if wait_match else (2**attempt)
                time.sleep(wait)
                continue
            return idx, f"[OCR ERROR on crop {idx}: {e}]"
    return idx, f"[OCR ERROR on crop {idx}: rate limit retries exhausted]"


def ocr_crops(crops):
    if not crops:
        return ""
    reader = get_easyocr_reader()
    try:
        client = get_groq_client()
    except ValueError:
        st.error("Please enter your Groq API key in the sidebar under 🔑 API Keys.")
        st.stop()
    n = len(crops)
    results = [""] * n

    progress = st.progress(0, text="Step 1/2 — Running EasyOCR draft…")
    easyocr_texts = [""] * n
    for i, crop in enumerate(crops):
        easyocr_texts[i] = _easyocr_single_crop(crop, reader)
        progress.progress((i + 1) / n, text=f"Step 1/2 — EasyOCR {i+1}/{n} crops…")
    progress.empty()

    progress = st.progress(0, text="Step 2/2 — Verifying with LLaMA-4 Scout…")
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        for i, crop in enumerate(crops):
            futures[
                ex.submit(_ocr_single_crop, (i, crop, client, easyocr_texts[i]))
            ] = i
            time.sleep(0.1)
        done = 0
        for fut in as_completed(futures):
            idx, text = fut.result()
            results[idx] = text
            done += 1
            progress.progress(
                done / n, text=f"Step 2/2 — LLaMA-4 Scout {done}/{n} crops…"
            )
    progress.empty()
    return "\n".join(t for t in results if t)


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: FIELD EXTRACTION (GEMINI)
# ══════════════════════════════════════════════════════════════════════
def extract_fields_llm(ocr_text, selected_fields):
    sys_p = (
        "You are an IT helpdesk document analyst. You extract structured fields from "
        "technical complaint and incident documents. The document may be a formal ticket, "
        "a printed form, or a narrative email / memo — fields may be presented as labelled "
        "key-value pairs OR buried inside paragraph text. Read the entire document carefully "
        "and infer field values from context when they are not explicitly labelled. "
        "Only use content explicitly present in the text. Use an empty string for any field "
        "that truly cannot be determined. Return ONLY valid JSON — no markdown fences, no preamble."
    )
    fields_list = "\n".join(f"{n+1}. {name}" for n, name in enumerate(selected_fields))
    user_p = f"Extract the following fields as a JSON object with these exact field names as keys:\n{fields_list}\n--- DOCUMENT START ---\n{ocr_text}\n--- DOCUMENT END ---"
    key = GEMINI_API_KEY
    if not key:
        return '{"error":"Gemini API key not configured on server."}'
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=sys_p,
        )
        response = model.generate_content(user_p)
        return response.text.strip()
    except Exception as e:
        return f'{{"error":"{str(e)}"}}'


# ══════════════════════════════════════════════════════════════════════
# SECTION 9: API KEY VALIDATION
# ══════════════════════════════════════════════════════════════════════
def validate_gemini_key(key):
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        model.generate_content("hi")
        return True, None
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            return True, None
        return False, err


def validate_groq_key(key):
    try:
        client = Groq(api_key=key)
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True, None
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            return True, None
        return False, err


# ══════════════════════════════════════════════════════════════════════
# SECTION 10: RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════
def parse_llm_json(raw):
    c = raw.strip()
    if c.startswith("```json"):
        c = c[7:].strip()
    elif c.startswith("```"):
        c = c[3:].strip()
    if c.endswith("```"):
        c = c[:-3].strip()
    c = c.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(c)
    except Exception:
        return None


def render_fields_table(fields, selected_fields):
    rows = ""
    for name in selected_fields:
        val = fields.get(name, "")
        if val:
            rows += f'<tr><td class="field-key">{name}</td><td class="field-val">{val}</td></tr>'
        else:
            rows += f'<tr><td class="field-key">{name}</td><td class="field-val field-empty">—</td></tr>'
    st.markdown(
        f'<div class="card"><table class="field-table">{rows}</table></div>',
        unsafe_allow_html=True,
    )


def render_ocr_text(text):
    lines = text.split("\n")
    html = []
    for line in lines:
        if line.startswith("[OCR ERROR"):
            html.append(f'<span class="ocr-error">{line}</span>')
        else:
            html.append(line)
    st.markdown(
        f'<div class="ocr-block">{"<br>".join(html)}</div>', unsafe_allow_html=True
    )


def fields_to_csv(fields, selected_fields):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Field", "Value"])
    for name in selected_fields:
        w.writerow([name, fields.get(name, "")])
    return buf.getvalue().encode("utf-8-sig")


def step_cls(done, active):
    if done:
        return "step done"
    if active:
        return "step active"
    return "step"


def convert_pdf_to_images(uploaded_file):
    import fitz

    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


# ══════════════════════════════════════════════════════════════════════
# SECTION 11: SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
<svg width="210" height="34" viewBox="0 0 680 104" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>.sb-logo-text { font-family: 'Segoe UI', sans-serif; }</style>
    <marker id="arr-sb" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#4f7bff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="8" y="10" width="62" height="80" rx="5" fill="#1e2330" opacity="0.5"/>
  <rect x="4" y="6"  width="62" height="80" rx="5" fill="#2a3050"/>
  <rect x="9" y="2"  width="62" height="80" rx="5" fill="#2e3a6e"/>
  <rect x="14" y="0" width="62" height="80" rx="5" fill="#1a2240" stroke="#4f7bff" stroke-width="0.8" opacity="0.9"/>
  <path d="M65 0 L76 11 L65 11 Z" fill="#243060"/>
  <path d="M65 0 L76 11" fill="none" stroke="#4f7bff" stroke-width="0.8" opacity="0.6"/>
  <line x1="22" y1="24" x2="60" y2="24" stroke="#4f7bff" stroke-width="2" stroke-linecap="round"/>
  <line x1="22" y1="35" x2="68" y2="35" stroke="#3d5299" stroke-width="1.2" stroke-linecap="round"/>
  <line x1="22" y1="45" x2="63" y2="45" stroke="#3d5299" stroke-width="1.2" stroke-linecap="round"/>
  <rect x="22" y="55" width="18" height="5" rx="1" fill="#4f7bff" opacity="0.3"/>
  <rect x="44" y="55" width="26" height="5" rx="1" fill="#4f7bff" opacity="0.2"/>
  <rect x="22" y="64" width="18" height="5" rx="1" fill="#4f7bff" opacity="0.3"/>
  <rect x="44" y="64" width="20" height="5" rx="1" fill="#4f7bff" opacity="0.2"/>
  <line x1="86" y1="40" x2="112" y2="40" stroke="#4f7bff" stroke-width="2" stroke-linecap="round" marker-end="url(#arr-sb)"/>
  <text x="122" y="56" class="sb-logo-text" font-size="52" font-weight="500" letter-spacing="-1"><tspan fill="#e2e8f0">Doc</tspan><tspan fill="#4f7bff">Struct</tspan></text>
</svg>
""",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.expander("🔑 Groq API Key", expanded=False):
        st.markdown(
            '<span class="sec-label">Bring your own keys</span>', unsafe_allow_html=True
        )
        st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Required — console.groq.com",
            key="ui_groq_key",
            help="Get a free key at console.groq.com",
        )

        if st.button("🔍 Validate Key", use_container_width=True):
            groq_key = st.session_state.get("ui_groq_key", "").strip()
            with st.spinner("Validating…"):
                results = {
                    "groq": (
                        validate_groq_key(groq_key)
                        if groq_key
                        else (None, "No key entered")
                    ),
                }
            st.session_state.key_validation = results

        if "key_validation" in st.session_state:
            for service, label in [("groq", "Groq")]:
                ok, err = st.session_state.key_validation.get(service, (None, ""))
                if ok is True:
                    st.markdown(
                        f'<div style="font-size:12px;padding:4px 0;">✅ <b>{label}</b> — valid</div>',
                        unsafe_allow_html=True,
                    )
                elif ok is False:
                    msg = err.split("\n")[0][:80] if err else "Invalid key"
                    st.markdown(
                        f'<div style="font-size:12px;padding:4px 0;color:#ef4444;">❌ <b>{label}</b> — {msg}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="font-size:12px;padding:4px 0;color:#94a3b8;">— <b>{label}</b> — {err}</div>',
                        unsafe_allow_html=True,
                    )

        st.caption("Your key is used only for your session and never stored.")

    st.markdown(
        '<span class="sec-label">Detection Controls</span>', unsafe_allow_html=True
    )
    conf_thresh = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.3, 0.05, key="conf_thresh"
    )
    iou_thresh = st.slider("IoU / NMS Threshold", 0.0, 1.0, 0.5, 0.05, key="iou_thresh")
    st.divider()

    with st.expander("📋 Fields to Extract", expanded=True):
        selected_fields = []
        for field_name in ALL_FIELD_NAMES:
            if st.checkbox(field_name, value=True, key=f"field_cb_{field_name}"):
                selected_fields.append(field_name)
    st.divider()

    st.markdown(
        '<span class="sec-label">Detection Summary</span>', unsafe_allow_html=True
    )
    stats_ph = st.empty()
    st.divider()

    proc_ph = st.empty()


# ══════════════════════════════════════════════════════════════════════
# SECTION 12: MAIN AREA — HEADER & FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
  <svg width="480" height="76" viewBox="0 0 680 108" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <style>.logo-text { font-family: 'Segoe UI', sans-serif; }</style>
      <marker id="arr-logo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M2 1L8 5L2 9" fill="none" stroke="#4f7bff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </marker>
    </defs>
    <rect x="8" y="10" width="62" height="80" rx="5" fill="#1e2330" opacity="0.5"/>
    <rect x="4" y="6"  width="62" height="80" rx="5" fill="#2a3050"/>
    <rect x="9"  y="2"  width="62" height="80" rx="5" fill="#2e3a6e"/>
    <rect x="14" y="0"  width="62" height="80" rx="5" fill="#1a2240" stroke="#4f7bff" stroke-width="0.8" opacity="0.9"/>
    <path d="M65 0 L76 11 L65 11 Z" fill="#243060"/>
    <path d="M65 0 L76 11" fill="none" stroke="#4f7bff" stroke-width="0.8" opacity="0.6"/>
    <line x1="22" y1="24" x2="60" y2="24" stroke="#4f7bff" stroke-width="2" stroke-linecap="round"/>
    <line x1="22" y1="35" x2="68" y2="35" stroke="#3d5299" stroke-width="1.2" stroke-linecap="round"/>
    <line x1="22" y1="45" x2="63" y2="45" stroke="#3d5299" stroke-width="1.2" stroke-linecap="round"/>
    <rect x="22" y="55" width="18" height="5" rx="1" fill="#4f7bff" opacity="0.3"/>
    <rect x="44" y="55" width="26" height="5" rx="1" fill="#4f7bff" opacity="0.2"/>
    <rect x="22" y="64" width="18" height="5" rx="1" fill="#4f7bff" opacity="0.3"/>
    <rect x="44" y="64" width="20" height="5" rx="1" fill="#4f7bff" opacity="0.2"/>
    <line x1="86" y1="40" x2="112" y2="40" stroke="#4f7bff" stroke-width="2" stroke-linecap="round" marker-end="url(#arr-logo)"/>
    <text x="122" y="56" class="logo-text" font-size="52" font-weight="500" letter-spacing="-1"><tspan fill="#e2e8f0">Doc</tspan><tspan fill="#4f7bff">Struct</tspan></text>
    <text x="123" y="78" class="logo-text" font-size="12" font-weight="400" letter-spacing="0.08em" fill="#6b7fa8">UNSTRUCTURED DOCS → STRUCTURED DATA</text>
  </svg>
  <div style="text-align:right; font-size:12px; color:#64748b; padding-top:4px;">
    🟢 easyocr + llama-4-scout (OCR) · gemini-2.5-flash (extraction) · roboflow
  </div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drop a document image or PDF here, or click to browse",
    type=["jpg", "jpeg", "png", "pdf"],
    label_visibility="visible",
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 13: FILE INGESTION & DETECTION
# ══════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    fkey = uploaded_file.name + str(uploaded_file.size)
    if st.session_state.get("last_fkey") != fkey:
        pages_data = []

        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Converting PDF pages to images…"):
                pil_images = convert_pdf_to_images(uploaded_file)
            for page_num, pil_img in enumerate(pil_images, 1):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    pil_img.save(tmp, format="PNG")
                    tmp_path = tmp.name
                try:
                    with st.spinner(
                        f"Running line detection on page {page_num}/{len(pil_images)}…"
                    ):
                        page_img = cv2.imread(tmp_path)
                        page_preds = run_inference(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                pages_data.append({"img": page_img, "predictions": page_preds})
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Running line detection…"):
                    page_img = cv2.imread(tmp_path)
                    page_preds = run_inference(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            pages_data.append({"img": page_img, "predictions": page_preds})

        st.session_state.last_fkey = fkey
        st.session_state.pages_data = pages_data
        st.session_state.img = pages_data[0]["img"]
        st.session_state.img_h = pages_data[0]["img"].shape[0]
        st.session_state.img_w = pages_data[0]["img"].shape[1]
        st.session_state.predictions = pages_data[0]["predictions"]
        for k in ("ocr_text", "llm_response", "fields_edited"):
            st.session_state.pop(k, None)
        st.rerun()

with proc_ph:
    process_btn = st.button(
        "▶ Process & Extract Fields",
        type="primary",
        use_container_width=True,
        disabled=("img" not in st.session_state),
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 14: STATE CHECKS & DETECTION STATS
# ══════════════════════════════════════════════════════════════════════
if "img" not in st.session_state:
    with stats_ph:
        st.markdown(
            """
        <div class="stat-row">
            <div class="stat-pill"><span class="stat-num" style="color:var(--muted)">—</span><span class="stat-lbl">Detected Lines</span></div>
        </div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="margin-top:50px;text-align:center;opacity:.3;">'
        '<div style="font-size:44px">📄</div>'
        '<div style="font-size:14px;margin-top:8px">Upload a document image or PDF above to begin</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

pages_data = st.session_state.get(
    "pages_data",
    [{"img": st.session_state.img, "predictions": st.session_state.predictions}],
)
num_pages = len(pages_data)
img = st.session_state.img
img_h = st.session_state.img_h
img_w = st.session_state.img_w
preds = st.session_state.predictions

per_page_detected = []
for pd in pages_data:
    p_img = pd["img"]
    p_det = postprocess(
        pd["predictions"], p_img.shape[1], p_img.shape[0], conf_thresh, iou_thresh
    )
    per_page_detected.append(p_det)

detected = per_page_detected[0]
all_regions = [{"box": e["box"], "type": "detected"} for e in detected]
total_lines = sum(len(d) for d in per_page_detected)

with stats_ph:
    st.markdown(
        f"""
    <div class="stat-row">
        <div class="stat-pill"><span class="stat-num" style="color:#10b981">{total_lines}</span><span class="stat-lbl">Detected Lines</span></div>
        <div class="stat-pill"><span class="stat-num" style="color:#3b82f6">{num_pages}</span><span class="stat-lbl">{'Page' if num_pages==1 else 'Pages'}</span></div>
    </div>""",
        unsafe_allow_html=True,
    )

has_ocr = "ocr_text" in st.session_state
has_fields = "llm_response" in st.session_state

st.markdown(
    f"""
<div class="steps">
  <div class="{step_cls(True,False)}"><div class="step-num">✓</div>Upload</div>
  <div class="{step_cls(True,False)}"><div class="step-num">✓</div>Detection</div>
  <div class="{step_cls(has_ocr, not has_ocr)}"><div class="step-num">{"✓" if has_ocr else "3"}</div>OCR</div>
  <div class="{step_cls(has_fields, has_ocr and not has_fields)}"><div class="step-num">{"✓" if has_fields else "4"}</div>Extraction</div>
</div>""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 15: PROCESS BUTTON HANDLER
# ══════════════════════════════════════════════════════════════════════
if process_btn:
    if not selected_fields:
        st.error("Please select at least one field to extract in the sidebar.")
    else:
        groq_key = st.session_state.get("ui_groq_key", "").strip()
        if not groq_key:
            st.error("Please enter your Groq API key in the sidebar under 🔑 API Keys.")
            st.stop()
        all_crops = []
        for page_idx, (pd, p_det) in enumerate(zip(pages_data, per_page_detected)):
            p_img = pd["img"]
            regions = [{"box": e["box"], "type": "detected"} for e in p_det]
            regions.sort(key=lambda r: (r["box"][1], r["box"][0]))
            all_crops.extend(crop_regions(p_img, regions))
        st.toast(
            f"Cropped {len(all_crops)} lines across {num_pages} page(s) — starting two-step OCR…"
        )
        ocr_text = ocr_crops(all_crops)
        st.session_state.ocr_text = ocr_text
        with st.spinner("Extracting fields with Gemini…"):
            llm_response = extract_fields_llm(ocr_text, selected_fields)
        if '"error":' in llm_response:
            st.error(f"Gemini extraction failed: {llm_response}")
            st.stop()
        st.session_state.llm_response = llm_response
        st.session_state.selected_fields_snapshot = list(selected_fields)
        parsed = parse_llm_json(llm_response)
        if parsed:
            st.session_state.fields_edited = {
                n: parsed.get(n, "") for n in selected_fields
            }
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# SECTION 16: OUTPUT TABS
# ══════════════════════════════════════════════════════════════════════
tab_images, tab_ocr, tab_fields = st.tabs(
    ["🔍 Image View", "📝 OCR Output", "🤖 Extracted Fields"]
)

with tab_images:
    for page_idx, (pd, p_det) in enumerate(zip(pages_data, per_page_detected)):
        p_img = pd["img"]
        if num_pages > 1:
            st.markdown(
                f'<span class="sec-label">Page {page_idx+1} of {num_pages}</span>',
                unsafe_allow_html=True,
            )
        img_disp = resize_for_display(p_img, max_long_edge=640)
        vis_disp = resize_for_display(draw_boxes(p_img, p_det), max_long_edge=640)
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.caption("📷 Original Document")
            st.image(to_display_bytes(img_disp), use_container_width=True)
        with col2:
            st.caption("🔍 Detection Overlay · 🟢 detected lines")
            st.image(to_display_bytes(vis_disp), use_container_width=True)
        page_regions = [{"box": e["box"], "type": "detected"} for e in p_det]
        crops_preview = crop_regions(p_img, page_regions)
        if crops_preview:
            st.markdown(
                f'<span class="sec-label" style="margin-top:12px;display:block">Crop Strip Preview — {len(crops_preview)} regions</span>',
                unsafe_allow_html=True,
            )
            n_show = min(len(crops_preview), 12)
            cols = st.columns(n_show, gap="small")
            for idx, (col, crop) in enumerate(zip(cols, crops_preview[:n_show])):
                with col:
                    h, w = crop.shape[:2]
                    thumb = cv2.resize(
                        crop,
                        (max(1, int(w * 40 / max(h, 1))), 40),
                        interpolation=cv2.INTER_AREA,
                    )
                    st.image(
                        to_display_bytes(thumb),
                        use_container_width=True,
                        caption=f"#{idx+1}",
                    )
            if len(crops_preview) > n_show:
                st.caption(f"… and {len(crops_preview)-n_show} more not shown")
        if num_pages > 1 and page_idx < num_pages - 1:
            st.divider()

with tab_ocr:
    if "ocr_text" in st.session_state:
        ocr_text = st.session_state.ocr_text
        hc1, hc2 = st.columns([4, 1])
        with hc1:
            st.markdown(f"**{len(ocr_text.splitlines())} lines extracted**")
        with hc2:
            st.download_button(
                "📥 .txt",
                data=ocr_text.encode("utf-8"),
                file_name="ocr_output.txt",
                mime="text/plain",
                use_container_width=True,
            )
        render_ocr_text(ocr_text)
        with st.expander("✏️ Edit OCR text & re-run extraction"):
            edited = st.text_area(
                "", ocr_text, height=300, label_visibility="collapsed"
            )
            if st.button("Re-run extraction with edited text"):
                active_fields = st.session_state.get(
                    "selected_fields_snapshot", selected_fields
                )
                with st.spinner("Re-running extraction…"):
                    new_resp = extract_fields_llm(edited, active_fields)
                st.session_state.ocr_text = edited
                st.session_state.llm_response = new_resp
                parsed = parse_llm_json(new_resp)
                if parsed:
                    st.session_state.fields_edited = {
                        n: parsed.get(n, "") for n in active_fields
                    }
                st.rerun()
    else:
        st.info("Press **▶ Process & Extract Fields** in the sidebar to run OCR.")

with tab_fields:
    if "llm_response" in st.session_state:
        active_fields = st.session_state.get(
            "selected_fields_snapshot", selected_fields
        )
        fields_edited = st.session_state.get("fields_edited")
        if fields_edited is None:
            parsed = parse_llm_json(st.session_state.llm_response)
            if parsed:
                fields_edited = {n: parsed.get(n, "") for n in active_fields}
                st.session_state.fields_edited = fields_edited
        if fields_edited:
            with st.expander("📋 Extracted Fields", expanded=True):
                render_fields_table(fields_edited, active_fields)
            with st.expander("✏️ Edit fields before export"):
                updated = {}
                ca, cb = st.columns(2)
                for idx, name in enumerate(active_fields):
                    col = ca if idx % 2 == 0 else cb
                    with col:
                        updated[name] = st.text_input(
                            name, value=fields_edited.get(name, ""), key=f"ef_{idx}"
                        )
                if st.button("💾 Save edits"):
                    st.session_state.fields_edited = updated
                    st.success("Edits saved.")
                    st.rerun()
            st.markdown("**Export**")
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.download_button(
                    "📥 JSON",
                    data=json.dumps(fields_edited, ensure_ascii=False, indent=2).encode(
                        "utf-8"
                    ),
                    file_name="extracted_fields.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with ec2:
                st.download_button(
                    "📥 CSV",
                    data=fields_to_csv(fields_edited, active_fields),
                    file_name="extracted_fields.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec3:
                st.download_button(
                    "📥 Raw LLM JSON",
                    data=st.session_state.llm_response.encode("utf-8"),
                    file_name="llm_raw.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.warning("Could not parse JSON from LLM response:")
            st.code(st.session_state.llm_response, language="json")
    else:
        st.info(
            "Press **▶ Process & Extract Fields** in the sidebar to run extraction."
        )

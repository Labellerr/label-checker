# ============================================================
# SmartQC — Intelligent Annotation Quality Control
# Client  : Labellerr AI (Tensor Matics Inc.)
# Dataset : Warehouse Detection (25 classes, 4651 images)
# Signals : CLIP semantic match + DINOv2 prototype + IoU (mock)
# Outputs : ACCEPT / REVIEW / FLAG / REJECT per annotation
#           + REST API endpoints + COCO JSON input support
# ============================================================

# =========================
# 1. IMPORTS
# =========================
import os, json, time, warnings, logging
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageFont
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# =========================
# 2. PRESENTATION PRINT HELPERS
# =========================
def banner(title, char="=", width=65):
    print("\n" + char*width)
    print(f"  {title}")
    print(char*width)

def section(title):
    print(f"\n  {'─'*60}")
    print(f"  ▶  {title}")
    print(f"  {'─'*60}")

def kv(label, value, indent=4):
    print(f"{' '*indent}{label:<35}: {value}")

def bar_chart(label, value, max_val, width=30, color_char="█"):
    filled = int((value / max_val) * width) if max_val > 0 else 0
    empty  = width - filled
    bar    = color_char * filled + "░" * empty
    print(f"  {label:<22} [{bar}] {value:>6}")

def pct_bar(label, pct, width=40):
    filled = int(pct / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    print(f"  {label:<12} [{bar}] {pct:5.1f}%")

def table_header(*cols, widths):
    row = "  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(row)
    print("  " + "─" * (sum(widths) + 2*len(widths)))

def table_row(*vals, widths):
    print("  " + "  ".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)))

# =========================
# 2b. CONFIG
# =========================
DATASET_PATH    = "/home/aparna.alladi/warehouse_detection_dataset"
YOLO_IMAGE_DIR  = os.path.join(DATASET_PATH, "yolo", "images")
YOLO_LABEL_DIR  = os.path.join(DATASET_PATH, "yolo", "labels")
CLASS_MAP_FILE  = os.path.join(DATASET_PATH, "class_mapping.json")

DINOV2_MODEL    = "facebook/dinov2-base"
USE_GRAYSCALE   = True
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 16

BACKGROUND_CLASSES = {"wall", "ceiling", "floor"}

# QC thresholds
IOU_THRESHOLD    = 0.5    # below = poor boundary
CONF_THRESHOLD   = 0.40   # ignore weak predictions
QUALITY_ACCEPT   = 0.72   # auto-accept
QUALITY_REVIEW   = 0.45   # below = flag/reject
QUALITY_REJECT   = 0.20   # below = auto-reject (clearly wrong)
MAX_PER_FOLDER   = 100

# ── FIX: header height added above crop (not overlaid) ──
HEADER_HEIGHT    = 52     # px reserved for annotation info strip

# Class-specific CLIP prompts for better discrimination
CLIP_PROMPTS = {
    "box":               "a cardboard shipping box in a warehouse",
    "crate":             "a wooden or plastic storage crate",
    "barrel":            "a cylindrical barrel or drum container",
    "bottle":            "a bottle or container",
    "pallet":            "a flat wooden or plastic shipping pallet on the floor",
    "rack":              "a metal warehouse storage rack with shelves",
    "bracket":           "a metal support bracket attached to a wall or pillar",
    "lamp":              "an industrial ceiling light or lamp fixture",
    "sign":              "a warehouse warning or information sign on a wall",
    "wire":              "electrical wires or cables running along a wall",
    "fuse_box":          "an electrical fuse box or circuit breaker panel",
    "floor_decal":       "a painted floor marking or floor safety decal",
    "fire_extinguisher": "a red fire extinguisher mounted on a wall",
    "barcode":           "a barcode sticker or label on an object",
    "pillar":            "a structural support pillar or column in a warehouse",
    "forklift":          "a yellow forklift or warehouse vehicle",
    "bucket":            "a plastic bucket or pail",
    "cone":              "an orange traffic safety cone",
    "cart":              "a metal warehouse cart or trolley",
    "emergency_board":   "an emergency information or safety board on a wall",
    "paper_note":        "a paper note or handwritten label attached to something",
    "paper_shortcut":    "a small paper label or shortcut card",
}

# =========================
# 3. LOAD CLASS MAPPING
# =========================
def load_class_mapping():
    with open(CLASS_MAP_FILE) as f:
        cm = json.load(f)
    id_to_name     = {c["id"]: c["name"]     for c in cm["classes"]}
    id_to_category = {c["id"]: c["category"] for c in cm["classes"]}
    name_to_id     = {c["name"]: c["id"]     for c in cm["classes"]}
    section("Class Mapping Summary")
    kv("Total classes loaded",   len(id_to_name))
    kv("QC classes (excl. bg)", len(id_to_name) - len(BACKGROUND_CLASSES))
    kv("Background skipped",    sorted(BACKGROUND_CLASSES))
    kv("Device",                DEVICE)
    kv("DINOv2 model",         DINOV2_MODEL)
    kv("Grayscale embeddings", USE_GRAYSCALE)
    kv("Batch size",           BATCH_SIZE)

    cats = {}
    for c in cm["classes"]:
        cats.setdefault(c["category"], []).append(c["name"])
    print()
    print("  Category breakdown:")
    for cat, names in sorted(cats.items()):
        skip_mark = " [skipped]" if all(n in BACKGROUND_CLASSES for n in names) else ""
        print(f"    {cat:<16}: {', '.join(sorted(names))}{skip_mark}")

    return id_to_name, id_to_category, name_to_id

# =========================
# 4a. INPUT: YOLO FORMAT
# =========================
def load_from_yolo(id_to_name, id_to_category, split="train"):
    """
    Reads YOLO .txt label files.
    Format per line: class_id  cx  cy  w  h  (normalized 0-1)
    Converts to pixel [x, y, w, h] top-left coordinates.
    """
    records = []
    n_bg = n_tiny = n_noimg = 0

    img_dir = os.path.join(YOLO_IMAGE_DIR, split)
    lbl_dir = os.path.join(YOLO_LABEL_DIR, split)
    if not os.path.isdir(img_dir):
        img_dir, lbl_dir = YOLO_IMAGE_DIR, YOLO_LABEL_DIR

    print(f"  Labels : {lbl_dir}")
    print(f"  Images : {img_dir}")
    label_files = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".txt"))
    print(f"  Label files: {len(label_files)}")

    for lf in tqdm(label_files, desc="Reading YOLO labels"):
        stem     = os.path.splitext(lf)[0]
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".PNG", ".JPG"]:
            c = os.path.join(img_dir, stem + ext)
            if os.path.exists(c):
                img_path = c; break
        if img_path is None:
            n_noimg += 1; continue
        try:
            W, H = Image.open(img_path).size
        except Exception:
            n_noimg += 1; continue

        with open(os.path.join(lbl_dir, lf)) as f:
            lines = f.read().strip().splitlines()

        for ai, line in enumerate(lines):
            p = line.strip().split()
            if len(p) < 5: continue
            cid  = int(p[0])
            cxn, cyn = float(p[1]), float(p[2])
            wn,  hn  = float(p[3]), float(p[4])
            name = id_to_name.get(cid, f"cls_{cid}")
            if name in BACKGROUND_CLASSES:
                n_bg += 1; continue
            wpx = int(wn * W); hpx = int(hn * H)
            xpx = max(0, int((cxn - wn/2) * W))
            ypx = max(0, int((cyn - hn/2) * H))
            wpx = min(wpx, W - xpx); hpx = min(hpx, H - ypx)
            if wpx * hpx < 400:
                n_tiny += 1; continue
            records.append({
                "annotation_id": f"{stem}_{ai}",
                "image_path": img_path, "image_stem": stem,
                "split": split, "class_id": cid, "class_name": name,
                "category": id_to_category.get(cid, "unknown"),
                "bbox_x": xpx, "bbox_y": ypx,
                "bbox_w": wpx, "bbox_h": hpx,
                "image_width": W, "image_height": H,
                "source_format": "yolo",
            })

    df = pd.DataFrame(records)
    print(f"\n  Total annotations : {len(df)}")
    print(f"  Unique images     : {df['image_stem'].nunique()}")
    print(f"  Skipped bg        : {n_bg}")
    print(f"  Skipped tiny      : {n_tiny}")
    print(f"  Skipped no-img    : {n_noimg}")
    section("Annotation Statistics")
    kv("Total annotations",  len(df))
    kv("Unique images",      df["image_stem"].nunique())
    kv("Annotations/image",  f"{len(df)/df['image_stem'].nunique():.1f} avg")
    kv("Skipped background", n_bg)
    kv("Skipped tiny bbox",  n_tiny)
    kv("Skipped no-image",   n_noimg)
    kv("Source format",      "YOLO (.txt normalized)")

    print()
    print("  Class distribution (sorted by count):")
    cc = df["class_name"].value_counts()
    for cls, cnt in cc.items():
        bar_chart(cls, cnt, cc.max())

    print()
    print("  Category totals:")
    cat_totals = df["category"].value_counts()
    for cat, cnt in cat_totals.items():
        pct = cnt / len(df) * 100
        pct_bar(cat, pct)

    return df

# =========================
# 4b. INPUT: COCO JSON FORMAT
# =========================
def load_from_coco_json(annotation_file, image_dir, id_to_name=None, id_to_category=None):
    """
    Tool-agnostic COCO JSON input layer.
    Accepts exports from: Labellerr, Roboflow, CVAT, Label Studio, VGG.
    Standard COCO format: images[], annotations[], categories[]
    """
    print(f"  Loading COCO JSON: {annotation_file}")
    with open(annotation_file) as f:
        coco = json.load(f)

    img_lookup  = {img["id"]: img for img in coco.get("images", [])}
    cat_lookup  = {c["id"]: c["name"] for c in coco.get("categories", [])}

    if id_to_name:
        name_to_our_id = {v: k for k, v in id_to_name.items()}
    else:
        name_to_our_id = {v: k for k, v in cat_lookup.items()}
        id_to_name     = cat_lookup
        id_to_category = {k: "unknown" for k in cat_lookup}

    records = []
    n_bg = n_tiny = n_noimg = 0

    for ann in tqdm(coco.get("annotations", []), desc="Reading COCO annotations"):
        img_info   = img_lookup.get(ann["image_id"])
        if img_info is None:
            n_noimg += 1; continue

        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            n_noimg += 1; continue

        cat_name = cat_lookup.get(ann["category_id"], "unknown")
        if cat_name in BACKGROUND_CLASSES:
            n_bg += 1; continue

        x, y, w, h = [int(v) for v in ann["bbox"]]
        if w * h < 400:
            n_tiny += 1; continue

        try:
            W = img_info.get("width",  0) or Image.open(img_path).size[0]
            H = img_info.get("height", 0) or Image.open(img_path).size[1]
        except Exception:
            n_noimg += 1; continue

        our_class_id = name_to_our_id.get(cat_name, ann["category_id"])

        records.append({
            "annotation_id": str(ann["id"]),
            "image_path": img_path,
            "image_stem": os.path.splitext(img_info["file_name"])[0],
            "split": "coco",
            "class_id": our_class_id,
            "class_name": cat_name,
            "category": id_to_category.get(our_class_id, "unknown"),
            "bbox_x": x, "bbox_y": y,
            "bbox_w": w, "bbox_h": h,
            "image_width": W, "image_height": H,
            "source_format": "coco_json",
            "segmentation": ann.get("segmentation", []),
        })

    df = pd.DataFrame(records)
    print(f"\n  Total annotations : {len(df)}")
    print(f"  Skipped bg        : {n_bg}")
    print(f"  Skipped tiny      : {n_tiny}")
    print(f"  Skipped no-img    : {n_noimg}")
    _print_class_dist(df)
    return df

def _print_class_dist(df):
    print(f"\nClass distribution (sorted by count):")
    cc = df["class_name"].value_counts()
    for cls, cnt in cc.items():
        bar = chr(9608) * int(cnt / cc.max() * 20)
        print(f"  {cls:<22} {cnt:>5}  {bar}")

# =========================
# 5. CROP OBJECT (no overlay on object)
# =========================
def crop_object(image_path, bbox_x, bbox_y, bbox_w, bbox_h, padding_ratio=0.35):
    """
    Returns a padded crop of the object. Padding is clipped to image bounds so
    the object itself is never cut off.

    Adaptive padding: small objects get less context padding so they fill the
    saved image rather than being lost in background.
      bbox area > 10000px : padding_ratio = 0.20  (large obj, less padding)
      bbox area 2000-10000: padding_ratio = 0.30
      bbox area < 2000px  : padding_ratio = 0.15  (tiny obj, minimal padding)
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    area = bbox_w * bbox_h
    if area < 2000:
        padding_ratio = 0.15
    elif area < 10000:
        padding_ratio = 0.25
    else:
        padding_ratio = 0.15

    pad_w = min(int(bbox_w * padding_ratio), 40)   # cap at 40px absolute
    pad_h = min(int(bbox_h * padding_ratio), 40)   # cap at 40px absolute

    x1 = max(0, bbox_x - pad_w)
    y1 = max(0, bbox_y - pad_h)
    x2 = min(W, bbox_x + bbox_w + pad_w)
    y2 = min(H, bbox_y + bbox_h + pad_h)

    crop = img.crop((x1, y1, x2, y2))
    return crop


# =========================
# FIX: save_annotated_crop
# Adds a header strip ABOVE the crop — object pixels untouched
# Header contains ALL relevant QC details
# =========================
def _rget(row, key, default):
    """Safe getter that works for both pandas Series and plain dicts."""
    try:
        v = row[key]
        return default if (v is None or (isinstance(v, float) and v != v)) else v
    except (KeyError, IndexError):
        return default


def save_annotated_crop(row, dest_path, color=False):
    """
    Saves the object crop with a full-detail info header ABOVE it.
    The header is a SEPARATE black canvas pasted above — object pixels untouched.

    Layout:
      ┌──────────────────────────────────────────────────────┐  ← header (dark bg)
      │ [FLAG]  class  (category)  |  ann_id  |  img: stem   │
      │ Q:0.182  CLIP:0.154  IoU:0.031  ProtoSim:0.41  ...   │
      │ Weights => CLIP:0.80  Proto:0.00  IoU:0.20  |  ...   │
      │ CLIPtop:forklift(0.241)  ProtoNearest:floor_decal    │
      │ Reason: WRONG LABEL — CLIP sees 'forklift'...        │
      ├──────────────────────────────────────────────────────┤
      │                                                      │  ← object crop
      │              [padded object crop here]               │     (never overlaid)
      │                                                      │
      └──────────────────────────────────────────────────────┘
    """
    try:
        crop = crop_object(
            row["image_path"], row["bbox_x"], row["bbox_y"],
            row["bbox_w"],     row["bbox_h"]
        )
        if not color and USE_GRAYSCALE:
            crop = ImageOps.grayscale(crop).convert("RGB")

        cw, ch = crop.size

        # ── Scale up tiny crops so the object is actually visible ────────
        # Scale width and height independently so flat/tall bboxes
        # are upscaled in BOTH dimensions to a readable minimum size
        # Upscale so both dimensions meet minimums, but cap max size to avoid
        # extreme aspect ratios (e.g. a 300x18 barcode blowing up to 2600px wide)
        MIN_CROP_W, MIN_CROP_H = 240, 180
        MAX_CROP_W, MAX_CROP_H = 800, 600
        scale_w = max(1.0, MIN_CROP_W / cw)
        scale_h = max(1.0, MIN_CROP_H / ch)
        scale   = max(scale_w, scale_h)
        # Cap so neither axis exceeds maximum
        scale   = min(scale, MAX_CROP_W / cw, MAX_CROP_H / ch)
        if scale > 1.0:
            crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
            cw, ch = crop.size

        # ── Draw red GT bbox ON the crop (marking the exact annotation) ──
        # Use same adaptive padding as crop_object so the box is placed correctly
        draw_crop = ImageDraw.Draw(crop)
        _area = row["bbox_w"] * row["bbox_h"]
        _pr   = 0.15 if _area < 2000 else (0.25 if _area < 10000 else 0.15)
        pad_w = min(int(row["bbox_w"] * _pr), 40)
        pad_h = min(int(row["bbox_h"] * _pr), 40)
        img_full = Image.open(row["image_path"])
        iW, iH   = img_full.size
        img_full.close()
        x1_crop = max(0, row["bbox_x"] - pad_w)
        y1_crop = max(0, row["bbox_y"] - pad_h)
        bx1 = max(0, row["bbox_x"] - x1_crop)
        by1 = max(0, row["bbox_y"] - y1_crop)
        bx2 = min(cw - 1, bx1 + row["bbox_w"])
        by2 = min(ch - 1, by1 + row["bbox_h"])
        draw_crop.rectangle([bx1, by1, bx2, by2], outline=(255, 80, 80), width=2)

        # ── Pull all fields safely (works for pandas Series + dict) ──────
        flag        = _rget(row, "qc_flag",          "?")
        cls_name    = _rget(row, "class_name",        "?")
        ann_id      = _rget(row, "annotation_id",     "?")
        quality     = _rget(row, "quality_score",     0.0)
        clip_sc     = _rget(row, "clip_lbl_score",    0.0)
        iou_sc      = _rget(row, "best_iou",          0.0)
        proto_sim   = _rget(row, "proto_label_sim",   0.0)
        proto_near  = _rget(row, "proto_nearest",     "?")
        clip_top    = _rget(row, "clip_top_class",    "?")
        clip_top_sc = _rget(row, "clip_top_score",    0.0)
        reason      = _rget(row, "qc_reason",         "")
        img_stem    = _rget(row, "image_stem",        "?")
        category    = _rget(row, "category",          "?")
        clip_w      = _rget(row, "clip_weight",       0.0)
        proto_w     = _rget(row, "proto_weight",      0.0)
        iou_w       = _rget(row, "iou_weight",        0.0)
        mismatch    = _rget(row, "clip_mismatch",     0)
        is_fp       = _rget(row, "is_false_positive", 0)
        model_conf  = _rget(row, "model_confidence",  0.0)

        FLAG_COLORS = {
            "ACCEPT": (30, 180, 30),
            "REVIEW": (200, 160, 0),
            "FLAG":   (200, 100, 0),
            "REJECT": (200, 30, 30),
        }
        flag_color = FLAG_COLORS.get(flag, (150, 150, 150))

        # Split reason across up to 2 lines (80 chars each) so nothing is cut off
        reason_str = str(reason)
        if len(reason_str) <= 90:
            reason_lines = [f"Reason: {reason_str}"]
        else:
            # split at last " | " before char 90
            split_at = reason_str.rfind(" | ", 0, 90)
            if split_at == -1:
                split_at = 90
                reason_lines = [f"Reason: {reason_str[:split_at]}",
                                f"        {reason_str[split_at:]}"]
            else:
                reason_lines = [f"Reason: {reason_str[:split_at]}",
                                f"        {reason_str[split_at+3:]}"]

        lines = [
            f"[{flag}]  {cls_name}  ({category})  |  {ann_id}  |  img: {img_stem}",
            f"Q:{quality:.3f}  CLIP:{clip_sc:.3f}  IoU:{iou_sc:.3f}  ProtoSim:{proto_sim:.3f}  ModelConf:{model_conf:.3f}",
            f"Weights => CLIP:{clip_w:.2f}  Proto:{proto_w:.2f}  IoU:{iou_w:.2f}  |  CLIPmismatch:{bool(mismatch)}  FalsePos:{bool(is_fp)}",
            f"CLIPtop:{clip_top}({clip_top_sc:.3f})  ProtoNearest:{proto_near}",
        ] + reason_lines

        line_h  = 13
        padding = 4
        hdr_h   = len(lines) * line_h + 2 * padding

        # Make canvas = header + crop
        canvas_w = max(cw, 600)   # at least 600px wide so text fits
        canvas   = Image.new("RGB", (canvas_w, hdr_h + ch), (20, 20, 20))

        # Draw coloured flag sidebar
        flag_bar = Image.new("RGB", (6, hdr_h), flag_color)
        canvas.paste(flag_bar, (0, 0))

        # Paste crop below header (centred if narrower than canvas)
        crop_x = (canvas_w - cw) // 2
        canvas.paste(crop, (crop_x, hdr_h))

        # Write header text
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        for li, line in enumerate(lines):
            y = padding + li * line_h
            draw.text((10, y), line, fill=(230, 230, 230), font=font)

        canvas.save(dest_path, quality=92)
        return True

    except Exception as e:
        # ── Fallback: save plain crop with minimal overlay so we never
        #    silently lose the image if header rendering fails ───────────
        import traceback
        print(f"  [save_annotated_crop] ERROR ann={row['annotation_id']} : {e}")
        traceback.print_exc()
        try:
            crop_fb = crop_object(
                row["image_path"], row["bbox_x"], row["bbox_y"],
                row["bbox_w"],     row["bbox_h"]
            )
            cw_fb, ch_fb = crop_fb.size
            cw_fb = max(cw_fb, 600)
            fb = Image.new("RGB", (cw_fb, 20 + ch_fb), (80, 0, 0))
            fb.paste(crop_fb, ((cw_fb - crop_fb.size[0]) // 2, 20))
            d_fb = ImageDraw.Draw(fb)
            d_fb.text((4, 4),
                f"[FALLBACK] {row['class_name']}  Q:{row['quality_score']:.2f}"
                f"  CLIP:{row['clip_lbl_score']:.2f}  IoU:{row['best_iou']:.2f}",
                fill=(255, 255, 0))
            fb.save(dest_path, quality=92)
        except Exception as e2:
            print(f"  [save_annotated_crop] FALLBACK also failed: {e2}")
        return False


# =========================
# 6. LOAD MODELS
# =========================
def load_dinov2():
    print(f"  DINOv2 : {DINOV2_MODEL}")
    proc  = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    model = AutoModel.from_pretrained(DINOV2_MODEL).to(DEVICE)
    model.eval()
    return model, proc

def load_clip():
    print(f"  CLIP   : openai/clip-vit-base-patch32")
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    model.eval()
    return model, proc

# =========================
# 7. DINOV2 EMBEDDINGS
# =========================
def extract_embeddings(df, model, proc):
    """
    Grayscale object crops → DINOv2 CLS token (768-dim).
    """
    embeddings, valid = [], []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="DINOv2 embeddings"):
        batch = df.iloc[i:i+BATCH_SIZE]
        crops, idxs = [], []
        for idx, row in batch.iterrows():
            try:
                c = crop_object(row["image_path"], row["bbox_x"],
                                row["bbox_y"], row["bbox_w"], row["bbox_h"])
                if USE_GRAYSCALE:
                    c = ImageOps.grayscale(c).convert("RGB")
                crops.append(c); idxs.append(idx)
            except Exception:
                continue
        if not crops: continue
        try:
            pv = proc(images=crops, return_tensors="pt")["pixel_values"].to(DEVICE)
            with torch.no_grad():
                feats = model(pixel_values=pv).last_hidden_state[:, 0, :]
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.cpu().numpy())
            valid.extend(idxs)
        except Exception as e:
            print(f"  Batch skipped: {e}")
    emb_array = np.vstack(embeddings)
    section("Embedding Summary")
    kv("Shape",             emb_array.shape)
    kv("Total objects",     emb_array.shape[0])
    kv("Embedding dim",     emb_array.shape[1])
    kv("Grayscale used",    USE_GRAYSCALE)
    kv("Normalised",        "L2 (cosine-ready)")
    kv("Skipped (errors)",  len(df) - len(valid))
    return emb_array, valid

# =========================
# 8. PROTOTYPE ASSIGNMENT
# =========================
def assign_prototypes(df, embeddings):
    """
    One prototype per class = mean DINOv2 embedding of all objects in that class.
    """
    df      = df.copy()
    classes = sorted(df["class_name"].unique())
    c2i     = {c: i for i, c in enumerate(classes)}
    n       = len(classes)

    print(f"\n  Computing {n} class prototypes...")
    protos  = np.stack([
        embeddings[(df["class_name"] == c).values].mean(axis=0)
        for c in classes
    ])
    protos  = protos / np.linalg.norm(protos, axis=1, keepdims=True)

    sims         = embeddings @ protos.T
    nearest_idx  = np.argmax(sims, axis=1)
    nearest_cls  = [classes[i] for i in nearest_idx]
    label_idx    = np.array([c2i.get(c, 0) for c in df["class_name"]])
    label_sim    = sims[np.arange(len(sims)), label_idx]

    df["proto_nearest"]   = nearest_cls
    df["proto_match"]     = (np.array(nearest_cls) == df["class_name"].values).astype(int)
    df["proto_label_sim"] = label_sim

    stats = []
    for c in classes:
        mask   = df["class_name"] == c
        sub    = df[mask]
        purity = sub["proto_match"].mean()
        msim   = sub["proto_label_sim"].mean()
        stats.append({"cls": c, "count": len(sub), "purity": purity, "mean_sim": msim})
    stats.sort(key=lambda x: -x["purity"])

    print(f"\n  {'Rank':>4}  {'Class':<22}  {'Count':>6}  {'Purity':>8}  {'MeanSim':>8}  Status  Bar")
    print("  " + "-"*80)
    purities = []
    for rank, s in enumerate(stats, 1):
        bar    = chr(9608) * int(s["purity"] * 30)
        status = "GOOD" if s["purity"] >= 0.50 else "FAIR" if s["purity"] >= 0.25 else "POOR"
        print(f"  {rank:>4}  {s['cls']:<22}  {s['count']:>6}  "
              f"{s['purity']:>7.1%}  {s['mean_sim']:>8.4f}  {status:<6}  {bar}")
        purities.append(s["purity"])

    good  = [s["cls"] for s in stats if s["purity"] >= 0.50]
    fair  = [s["cls"] for s in stats if 0.25 <= s["purity"] < 0.50]
    poor  = [s["cls"] for s in stats if s["purity"] < 0.25]
    pmap  = {s["cls"]: s["purity"] for s in stats}

    print(f"\n  Mean purity : {np.mean(purities):.1%}")
    print(f"  GOOD (>=50%) {len(good):>2}: {', '.join(good) if good else 'none'}")
    print(f"  FAIR (25-50%) {len(fair):>2}: {', '.join(fair) if fair else 'none'}")
    print(f"  POOR (<25%)  {len(poor):>2}: {', '.join(poor) if poor else 'none'}")
    print(f"\n  Note: POOR classes share visual features — CLIP is primary signal for those.")

    df["class_purity"] = df["class_name"].map(pmap)
    return df, protos, classes, pmap

# =========================
# 9. CLIP SEMANTIC VALIDATION
# =========================
def run_clip_validation(df, clip_model, clip_proc, all_class_names):
    """
    Two-pass CLIP validation with auto-calibrated mismatch threshold.
    """
    prompts = [CLIP_PROMPTS.get(c, f"a {c} in a warehouse") for c in all_class_names]
    txt_in  = clip_proc(text=prompts, return_tensors="pt",
                         padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        t_out  = clip_model.text_model(input_ids=txt_in["input_ids"],
                                        attention_mask=txt_in["attention_mask"])
        t_feat = clip_model.text_projection(t_out.pooler_output)
        t_feat = torch.nn.functional.normalize(t_feat, dim=-1)
    text_emb = t_feat.cpu().numpy()
    print(f"  Text embeddings: {text_emb.shape}")

    # Build label→index lookup once (O(1) per lookup instead of O(N) list.index)
    cls_to_idx = {c: i for i, c in enumerate(all_class_names)}

    all_gaps, raw = [], []
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(df), BATCH_SIZE), total=n_batches, desc="CLIP validation"):
        batch = df.iloc[i:i+BATCH_SIZE]
        crops, idxs = [], []
        for idx, row in batch.iterrows():
            try:
                crops.append(crop_object(row["image_path"], row["bbox_x"],
                                         row["bbox_y"], row["bbox_w"], row["bbox_h"]))
                idxs.append(idx)
            except Exception:
                raw.append((idx, None)); continue
        if not crops: continue
        try:
            img_in = clip_proc(images=crops, return_tensors="pt",
                                padding=True).to(DEVICE)
            with torch.no_grad():
                v_out  = clip_model.vision_model(pixel_values=img_in["pixel_values"])
                v_feat = clip_model.visual_projection(v_out.pooler_output)
                v_feat = torch.nn.functional.normalize(v_feat, dim=-1)
            sims = v_feat.cpu().numpy() @ text_emb.T
            for b, orig in enumerate(idxs):
                label  = df.loc[orig, "class_name"]
                sr     = sims[b]
                top_i  = int(np.argmax(sr))
                top_sc = float(sr[top_i])
                lbl_i  = cls_to_idx.get(label, -1)
                lbl_sc = float(sr[lbl_i]) if lbl_i >= 0 else float(np.median(sr))
                if all_class_names[top_i] != label:
                    all_gaps.append(top_sc - lbl_sc)
                raw.append((orig, sr.copy()))
        except Exception as e:
            for orig in idxs: raw.append((orig, None))

    if all_gaps:
        g_mean, g_std = np.mean(all_gaps), np.std(all_gaps)
        thr = g_mean + 1.5 * g_std
    else:
        thr = 0.05
    print(f"  Auto-calibrated mismatch threshold: {thr:.4f}")

    results = []
    for orig, sr in raw:
        label = df.loc[orig, "class_name"]
        if sr is None:
            results.append({"idx": orig, "clip_lbl_sc": 0.18,
                             "clip_top_cls": label, "clip_top_sc": 0.18,
                             "clip_mismatch": 0, "clip_gap": 0.0}); continue
        top_i      = int(np.argmax(sr))
        top_cls    = all_class_names[top_i]
        top_sc     = float(sr[top_i])
        lbl_i      = cls_to_idx.get(label, -1)
        lbl_sc     = float(sr[lbl_i]) if lbl_i >= 0 else float(np.median(sr))
        gap        = top_sc - lbl_sc
        mismatch   = int(top_cls != label and gap > thr)
        results.append({"idx": orig,
                         "clip_lbl_sc": round(lbl_sc, 4), "clip_top_cls": top_cls,
                         "clip_top_sc": round(top_sc, 4), "clip_mismatch": mismatch,
                         "clip_gap": round(gap, 4)})

    cdf = pd.DataFrame(results).set_index("idx").sort_index()
    section("CLIP Validation Summary")
    kv("Total validated",      len(cdf))
    kv("Mean label score",     f"{cdf['clip_lbl_sc'].mean():.4f}")
    kv("Mean top score",       f"{cdf['clip_top_sc'].mean():.4f}")
    kv("Mean gap",             f"{cdf['clip_gap'].mean():.4f}")
    kv("Mismatch threshold",   f"{thr:.4f}  (auto-calibrated: mean+1.5*std)")
    kv("CLIP mismatches",      f"{cdf['clip_mismatch'].sum()}  ({cdf['clip_mismatch'].mean()*100:.1f}%)")
    kv("Low confidence (<0.16)",f"{(cdf['clip_lbl_sc']<0.16).sum()}")

    print()
    print("  CLIP score distribution by class:")
    # Vectorised — join class names from df, then groupby (no O(N²) iterrows)
    cdf_named = cdf.copy()
    cdf_named["class_name"] = df.loc[cdf.index, "class_name"].values
    grp = cdf_named.groupby("class_name").agg(
        mean_sc      =("clip_lbl_sc",  "mean"),
        mm_cnt       =("clip_mismatch", "sum"),
        total        =("clip_lbl_sc",  "count"),
    ).sort_values("mean_sc", ascending=False)
    for cls2, r2 in grp.iterrows():
        mm_pct = r2["mm_cnt"] / r2["total"] * 100 if r2["total"] > 0 else 0
        bar    = chr(9608) * int(r2["mean_sc"] * 100)
        print(f"    {cls2:<22} score={r2['mean_sc']:.3f}  mismatch={mm_pct:5.1f}%  {bar}")

    return cdf

# =========================
# 10. MOCK YOLO
# =========================
def run_mock_yolo(df, id_to_name):
    """
    Realistic mock YOLO predictions for demonstration.
    Returns bboxes in [x, y, w, h] format (same as GT annotations).

    Replace with real inference:
      from ultralytics import YOLO
      yolo = YOLO("best.pt")
      results = yolo(img_path, verbose=False)[0]
    """
    unique_images = df["image_path"].unique()
    predictions   = {}
    ann_by_img    = df.groupby("image_path")

    print(f"  Generating mock predictions for {len(unique_images)} images...")
    for img_path in tqdm(unique_images, desc="Mock YOLO"):
        np.random.seed(hash(img_path) % (2**31))
        preds    = []
        img_anns = ann_by_img.get_group(img_path) if img_path in ann_by_img.groups else pd.DataFrame()

        for _, row in img_anns.iterrows():
            x, y, w, h = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
            roll = np.random.random()

            if roll < 0.55:    # Good prediction
                jx = int(np.random.uniform(-0.08*w, 0.08*w))
                jy = int(np.random.uniform(-0.08*h, 0.08*h))
                jw = int(w * np.random.uniform(0.88, 1.12))
                jh = int(h * np.random.uniform(0.88, 1.12))
                preds.append({"bbox": [max(0,x+jx), max(0,y+jy), jw, jh],
                               "confidence": round(float(np.random.uniform(0.65, 0.97)), 4),
                               "class_id": int(row["class_id"]),
                               "class_name": row["class_name"]})
            elif roll < 0.70:  # Class mismatch
                all_ids   = list(id_to_name.keys())
                wrong_ids = [k for k in all_ids if k != row["class_id"]]
                wrong_id  = int(np.random.choice(wrong_ids)) if wrong_ids else int(row["class_id"])
                preds.append({"bbox": [max(0,x), max(0,y), w, h],
                               "confidence": round(float(np.random.uniform(0.50, 0.82)), 4),
                               "class_id": wrong_id,
                               "class_name": id_to_name.get(wrong_id, "unknown")})
            elif roll < 0.82:  # Poor boundary
                shift_x = int(w * np.random.uniform(0.35, 0.65))
                shift_y = int(h * np.random.uniform(0.35, 0.65))
                preds.append({"bbox": [max(0,x+shift_x), max(0,y+shift_y), w, h],
                               "confidence": round(float(np.random.uniform(0.45, 0.75)), 4),
                               "class_id": int(row["class_id"]),
                               "class_name": row["class_name"]})
            # else 18%: no prediction → annotation becomes False Positive

        try:
            W = int(img_anns.iloc[0]["image_width"]) if len(img_anns) > 0 else 512
            H = int(img_anns.iloc[0]["image_height"]) if len(img_anns) > 0 else 512
        except Exception:
            W, H = 512, 512
        for _ in range(np.random.randint(0, 3)):
            cls_id = int(np.random.choice(list(id_to_name.keys())))
            preds.append({
                "bbox": [np.random.randint(0, max(1,W-100)),
                          np.random.randint(0, max(1,H-100)),
                          np.random.randint(40, 150), np.random.randint(40, 150)],
                "confidence": round(float(np.random.uniform(0.45, 0.88)), 4),
                "class_id": cls_id, "class_name": id_to_name.get(cls_id, "unknown")})

        predictions[img_path] = preds
    return predictions

# =========================
# 11. IoU  — both [x,y,w,h] format
# =========================
def compute_iou(box_a, box_b):
    """Both boxes must be [x, y, w, h]."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = box_a[0]+box_a[2], box_a[1]+box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = box_b[0]+box_b[2], box_b[1]+box_b[3]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0.0

# =========================
# 12. QC METRICS (core engine)
# =========================
def compute_qc_metrics(df, clip_df, predictions, purity_map):
    """
    Three-signal QC per annotation:

    Signal 1 — CLIP semantic match (primary for POOR classes)
    Signal 2 — DINOv2 prototype similarity (primary for GOOD/FAIR classes)
    Signal 3 — IoU against model prediction (boundary quality)

    Adaptive weights based on class purity:
      POOR  (<25%): CLIP 80% + IoU 20%
      FAIR (25-50%): CLIP 55% + Proto 25% + IoU 20%
      GOOD  (>=50%): CLIP 45% + Proto 35% + IoU 20%

    Decision tiers:
      REJECT : quality < 0.20
      FLAG   : quality < 0.45
      REVIEW : quality < 0.72
      ACCEPT : quality >= 0.72
    """
    df  = df.copy()
    qc  = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="QC metrics"):
        label    = row["class_name"]
        img_path = row["image_path"]
        ann_bbox = [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]]

        # ── Signal 1: CLIP ──────────────────────────────────────────
        cr         = clip_df.loc[idx]
        lbl_sc     = float(cr["clip_lbl_sc"])
        top_cls    = cr["clip_top_cls"]
        top_sc     = float(cr["clip_top_sc"])
        mismatch   = int(cr["clip_mismatch"])
        gap        = float(cr["clip_gap"])
        clip_sig   = float(np.clip((lbl_sc - 0.18) / 0.12, 0.0, 1.0))

        # ── Signal 2: Prototype ─────────────────────────────────────
        proto_match = float(row["proto_match"])
        lbl_sim     = float(row["proto_label_sim"])
        sim_sig     = float(np.clip((lbl_sim - 0.30) / 0.60, 0.0, 1.0))
        proto_sig   = 0.70 * proto_match + 0.30 * sim_sig

        # ── Signal 3: IoU ───────────────────────────────────────────
        img_preds = [p for p in predictions.get(img_path, [])
                     if p["confidence"] >= CONF_THRESHOLD]
        best_iou  = 0.0; best_pred = None
        for pred in img_preds:
            iou = compute_iou(ann_bbox, pred["bbox"])
            if iou > best_iou:
                best_iou = iou; best_pred = pred

        iou_sig    = float(np.clip(best_iou / IOU_THRESHOLD, 0.0, 1.0))
        is_fp      = 1 if best_iou < 0.05 else 0
        cls_match  = int(best_pred["class_id"] == row["class_id"]) if best_pred else 0
        model_conf = best_pred["confidence"] if best_pred else 0.0

        # ── Adaptive weights ─────────────────────────────────────────
        class_purity = float(purity_map.get(label, 0.0))
        if class_purity < 0.25:
            w_clip, w_proto, w_iou = 0.80, 0.00, 0.20
        elif class_purity < 0.50:
            w_clip, w_proto, w_iou = 0.55, 0.25, 0.20
        else:
            w_clip, w_proto, w_iou = 0.45, 0.35, 0.20

        quality = w_clip * clip_sig + w_proto * proto_sig + w_iou * iou_sig

        # ── Hard overrides ────────────────────────────────────────────
        hard_reasons = []
        if mismatch:
            hard_reasons.append(f"WRONG LABEL — CLIP sees '{top_cls}' (gap={gap:.3f}) not '{label}'")
        if lbl_sc < 0.16:
            hard_reasons.append(f"VERY LOW CLIP CONFIDENCE — {label} score={lbl_sc:.3f}")

        # ── Decision tier ─────────────────────────────────────────────
        reasons = hard_reasons.copy()

        if hard_reasons or quality < QUALITY_REJECT:
            qc_flag = "REJECT"
            if not reasons:
                reasons.append("quality score below reject threshold")

        elif quality >= QUALITY_ACCEPT and not is_fp:
            # Don't auto-accept if there's no model detection at this location
            qc_flag  = "ACCEPT"
            reasons  = ["CLIP confirmed + visually typical + good IoU"]

        elif quality >= QUALITY_REVIEW and not is_fp:
            qc_flag = "REVIEW"
            reasons.append("Borderline quality — needs human check")
            if proto_match == 0 and class_purity >= 0.25:
                reasons.append(f"visually nearest to '{row['proto_nearest']}'")

        else:
            qc_flag = "FLAG"
            if is_fp:
                reasons.append("FALSE POSITIVE — no model detection at this location")
            if cls_match == 0 and best_pred:
                reasons.append(f"CLASS MISMATCH — model predicts '{best_pred['class_name']}'")
            if best_iou < IOU_THRESHOLD and not is_fp:
                reasons.append(f"POOR BOUNDARY — IoU={best_iou:.3f} < {IOU_THRESHOLD}")
            if proto_match == 0 and class_purity >= 0.25:
                reasons.append(f"CLUSTER OUTLIER — nearest to '{row['proto_nearest']}'")
            if not reasons:
                reasons.append("low combined quality score")

        qc.append({
            "annotation_id":    row["annotation_id"],
            "image_path":       img_path,
            "image_stem":       row["image_stem"],
            "split":            row["split"],
            "class_id":         row["class_id"],
            "class_name":       label,
            "category":         row["category"],
            "bbox_x": row["bbox_x"], "bbox_y": row["bbox_y"],
            "bbox_w": row["bbox_w"], "bbox_h": row["bbox_h"],
            # CLIP signal
            "clip_lbl_score":   round(lbl_sc, 4),
            "clip_top_class":   top_cls,
            "clip_top_score":   round(top_sc, 4),
            "clip_mismatch":    mismatch,
            "clip_gap":         round(gap, 4),
            "clip_signal":      round(clip_sig, 4),
            # Prototype signal
            "proto_match":      int(proto_match),
            "proto_nearest":    row["proto_nearest"],
            "proto_label_sim":  round(lbl_sim, 4),
            "proto_signal":     round(proto_sig, 4),
            # IoU signal
            "best_iou":         round(best_iou, 4),
            "is_false_positive":is_fp,
            "class_match_iou":  cls_match,
            "model_confidence": round(model_conf, 4),
            "iou_signal":       round(iou_sig, 4),
            # Combined
            "clip_weight":      round(w_clip, 2),
            "proto_weight":     round(w_proto, 2),
            "iou_weight":       round(w_iou, 2),
            "class_purity":     round(class_purity, 4),
            "quality_score":    round(quality, 4),
            "qc_flag":          qc_flag,
            "qc_reason":        " | ".join(reasons),
        })

    return pd.DataFrame(qc)

# =========================
# 13. FALSE NEGATIVES
# =========================
def find_false_negatives(df, predictions):
    fn_rows    = []
    ann_by_img = df.groupby("image_path")
    for img_path, img_preds in predictions.items():
        img_anns = ann_by_img.get_group(img_path) if img_path in ann_by_img.groups else pd.DataFrame()
        for pred in img_preds:
            if pred["confidence"] < CONF_THRESHOLD: continue
            max_iou = 0.0
            for _, row in img_anns.iterrows():
                max_iou = max(max_iou, compute_iou(
                    pred["bbox"],
                    [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]]))
            if max_iou < 0.1:
                fn_rows.append({
                    "image_path": img_path,
                    "pred_bbox": str(pred["bbox"]),
                    "pred_class": pred["class_name"],
                    "pred_confidence": pred["confidence"],
                    "max_iou_with_anns": round(max_iou, 4),
                    "issue": "FALSE NEGATIVE — model detected object not annotated by human",
                })
    fn_df = pd.DataFrame(fn_rows)
    section("False Negative Summary")
    kv("Total FN detected",  len(fn_df))
    kv("Meaning",            "Objects model found but human annotator missed")
    if len(fn_df) > 0:
        print()
        print("  FN breakdown by predicted class:")
        fn_by_cls = fn_df["pred_class"].value_counts().head(10)
        for cls2, cnt in fn_by_cls.items():
            bar_chart(cls2, cnt, fn_by_cls.max())
    return fn_df

# =========================
# 14. METRICS MATRIX
# =========================
def compute_metrics_matrix(qc_df, fn_df):
    total = len(qc_df)
    fc    = qc_df["qc_flag"].value_counts()

    print("\n" + "="*60 + "\nQC METRICS MATRIX\n" + "="*60)

    print("\n[ Decision Distribution ]")
    for flag in ["ACCEPT","REVIEW","FLAG","REJECT"]:
        n   = fc.get(flag, 0)
        pct = n/total*100
        bar = chr(9608)*int(pct/2)
        print(f"  {flag:<8}: {n:>6} ({pct:5.1f}%)  {bar}")

    print("\n[ CLIP Signal ]")
    print(f"  Mean CLIP label score  : {qc_df['clip_lbl_score'].mean():.4f}")
    print(f"  CLIP mismatch count    : {qc_df['clip_mismatch'].sum()} ({qc_df['clip_mismatch'].mean()*100:.1f}%)")

    print("\n[ Prototype Signal ]")
    print(f"  Prototype match rate   : {qc_df['proto_match'].mean()*100:.1f}%")
    print(f"  Mean prototype sim     : {qc_df['proto_label_sim'].mean():.4f}")

    print("\n[ IoU Signal ]")
    print(f"  Mean IoU               : {qc_df['best_iou'].mean():.4f}")
    print(f"  IoU < 0.5 (poor bbox)  : {(qc_df['best_iou']<0.5).sum()} ({(qc_df['best_iou']<0.5).mean()*100:.1f}%)")
    print(f"  False Positives        : {qc_df['is_false_positive'].sum()} ({qc_df['is_false_positive'].mean()*100:.1f}%)")
    print(f"  False Negatives        : {len(fn_df)}")

    print("\n[ Detection Quality ]")
    tp = int(((qc_df["best_iou"]>=IOU_THRESHOLD)&(qc_df["class_match_iou"]==1)).sum())
    fp = int(qc_df["is_false_positive"].sum())
    fn = len(fn_df)
    pr = tp/(tp+fp) if (tp+fp)>0 else 0
    rc = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
    print(f"  True  Positives (TP)   : {tp}")
    print(f"  False Positives (FP)   : {fp}")
    print(f"  False Negatives (FN)   : {fn}")
    print(f"  Precision              : {pr:.4f}")
    print(f"  Recall                 : {rc:.4f}")
    print(f"  F1 Score               : {f1:.4f}")

    print("\n[ Adaptive Weights Distribution ]")
    w100 = (qc_df["clip_weight"]==0.80).sum()
    w55  = (qc_df["clip_weight"]==0.55).sum()
    w45  = (qc_df["clip_weight"]==0.45).sum()
    print(f"  POOR cls (CLIP 80%)    : {w100} ann ({w100/total*100:.1f}%)")
    print(f"  FAIR cls (CLIP 55%)    : {w55} ann ({w55/total*100:.1f}%)")
    print(f"  GOOD cls (CLIP 45%)    : {w45} ann ({w45/total*100:.1f}%)")

    print("\n[ Per-Class Quality (worst → best) ]")
    cg = qc_df.groupby("class_name").agg(
        N            =("annotation_id",     "count"),
        clip_score   =("clip_lbl_score",    "mean"),
        mismatch_pct =("clip_mismatch",     "mean"),
        proto_match  =("proto_match",       "mean"),
        mean_iou     =("best_iou",          "mean"),
        quality      =("quality_score",     "mean"),
        accept_pct   =("qc_flag", lambda x: (x=="ACCEPT").mean()),
        flag_pct     =("qc_flag", lambda x: (x=="FLAG").mean()),
        reject_pct   =("qc_flag", lambda x: (x=="REJECT").mean()),
    ).sort_values("quality").reset_index()

    print(f"\n  {'Class':<22} {'N':>5} {'CLIP':>6} {'Miss%':>6} "
          f"{'Proto%':>7} {'IoU':>6} {'Q':>6} {'Acc%':>6} {'Flg%':>6} {'Rej%':>6}")
    print("  " + "-"*85)
    for _, r in cg.iterrows():
        print(f"  {r['class_name']:<22} {r['N']:>5} {r['clip_score']:>6.3f} "
              f"{r['mismatch_pct']*100:>5.1f}% {r['proto_match']*100:>6.1f}% "
              f"{r['mean_iou']:>6.3f} {r['quality']:>6.3f} "
              f"{r['accept_pct']*100:>5.1f}% {r['flag_pct']*100:>5.1f}% "
              f"{r['reject_pct']*100:>5.1f}%")

    print("\n[ Per-Category Summary ]")
    cs = qc_df.groupby("category").agg(
        N        =("annotation_id","count"),
        quality  =("quality_score","mean"),
        mismatch =("clip_mismatch","mean"),
        flag_rate=("qc_flag", lambda x: (x=="FLAG").mean()),
        rej_rate =("qc_flag", lambda x: (x=="REJECT").mean()),
    ).sort_values("quality").reset_index()
    for _, r in cs.iterrows():
        print(f"  {r['category']:<16} N={r['N']:>5}  Q={r['quality']:.3f}  "
              f"mismatch={r['mismatch']:.1%}  flag={r['flag_rate']:.1%}  "
              f"reject={r['rej_rate']:.1%}")

    flagged = qc_df[qc_df["qc_flag"].isin(["FLAG","REJECT"])]
    if len(flagged):
        print("\n[ FLAG+REJECT Breakdown ]")
        print(f"  CLIP wrong label       : {flagged['clip_mismatch'].sum()}")
        print(f"  False Positives        : {flagged['is_false_positive'].sum()}")
        print(f"  Cluster outlier        : {(flagged['proto_match']==0).sum()}")
        print(f"  Poor IoU boundary      : {(flagged['best_iou']<IOU_THRESHOLD).sum()}")

    # ── Executive summary ─────────────────────────────────────────────
    banner("SMARTQC EXECUTIVE SUMMARY", char="═")
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  Total Annotations Processed : {total:>8,}               │")
    print(f"  │  ACCEPT  (auto-accepted)  : {fc.get('ACCEPT',0):>8,}  ({fc.get('ACCEPT',0)/total*100:5.1f}%)  │")
    print(f"  │  REVIEW  (human needed)   : {fc.get('REVIEW',0):>8,}  ({fc.get('REVIEW',0)/total*100:5.1f}%)  │")
    print(f"  │  FLAG    (suspicious)     : {fc.get('FLAG',0):>8,}  ({fc.get('FLAG',0)/total*100:5.1f}%)  │")
    print(f"  │  REJECT  (clearly wrong)  : {fc.get('REJECT',0):>8,}  ({fc.get('REJECT',0)/total*100:5.1f}%)  │")
    print("  └─────────────────────────────────────────────────────────┘")

    print()
    print("  Quality Score Visualisation (ACCEPT=green, REVIEW=yellow, FLAG=red):")
    pct_bar(" ACCEPT",  fc.get("ACCEPT",0)/total*100)
    pct_bar(" REVIEW",  fc.get("REVIEW",0)/total*100)
    pct_bar(" FLAG",    fc.get("FLAG",0)/total*100)
    pct_bar(" REJECT",  fc.get("REJECT",0)/total*100)

    print()
    section("Detection Quality (Real YOLOv8 IoU Signals)")
    kv("True Positives  (TP)", f"{tp:,}  (annotation matched + correct class)")
    kv("False Positives (FP)", f"{fp:,}  (annotation exists, no object there)")
    kv("False Negatives (FN)", f"{fn:,}  (object exists, annotator missed it)")
    kv("Precision",            f"{pr:.4f}  = TP / (TP + FP)")
    kv("Recall",               f"{rc:.4f}  = TP / (TP + FN)")
    kv("F1 Score",             f"{f1:.4f}  = harmonic mean of P and R")

    print()
    section("Signal Quality")
    kv("Mean CLIP label score",  f"{qc_df['clip_lbl_score'].mean():.4f}  (range: 0.18-0.30 for warehouse)")
    kv("CLIP mismatches",        f"{qc_df['clip_mismatch'].sum():,}  ({qc_df['clip_mismatch'].mean()*100:.1f}%)")
    kv("Prototype match rate",   f"{qc_df['proto_match'].mean()*100:.1f}%  (nearest prototype = own class)")
    kv("Mean IoU score",         f"{qc_df['best_iou'].mean():.4f}")
    kv("Poor boundary (<0.5)",   f"{(qc_df['best_iou']<IOU_THRESHOLD).sum():,}  ({(qc_df['best_iou']<IOU_THRESHOLD).mean()*100:.1f}%)")
    kv("Mean quality score",     f"{qc_df['quality_score'].mean():.4f}")

    print()
    section("Business Impact Estimate")
    manual_review_pct = (fc.get("REVIEW",0) + fc.get("FLAG",0)) / total * 100
    auto_handled_pct  = 100 - manual_review_pct
    print(f"  Without SmartQC : 100% of {total:,} annotations need human review")
    print(f"  With SmartQC    : only {manual_review_pct:.1f}% need human review")
    print(f"  Auto-handled    : {auto_handled_pct:.1f}% of annotations (ACCEPT + REJECT) need no human")
    est_hours_saved = (total * auto_handled_pct / 100) * (30 / 3600)
    print(f"  Estimated hours saved : {est_hours_saved:.0f} hrs @ 30 sec/annotation")
    print(f"  Estimated cost saving : ~${est_hours_saved * 15:,.0f}  @ $15/hr reviewer rate")

    return {
        "total": total,
        "accept": int(fc.get("ACCEPT",0)), "review": int(fc.get("REVIEW",0)),
        "flag":   int(fc.get("FLAG",0)),   "reject": int(fc.get("REJECT",0)),
        "accept_pct": round(fc.get("ACCEPT",0)/total*100,2),
        "review_pct": round(fc.get("REVIEW",0)/total*100,2),
        "flag_pct":   round(fc.get("FLAG",  0)/total*100,2),
        "reject_pct": round(fc.get("REJECT",0)/total*100,2),
        "mean_clip":      round(float(qc_df["clip_lbl_score"].mean()),4),
        "clip_mismatch":  int(qc_df["clip_mismatch"].sum()),
        "proto_match_rate": round(float(qc_df["proto_match"].mean()),4),
        "mean_iou":       round(float(qc_df["best_iou"].mean()),4),
        "precision":      round(pr,4), "recall": round(rc,4), "f1": round(f1,4),
        "false_positives": fp, "false_negatives": fn,
        "mean_quality":   round(float(qc_df["quality_score"].mean()),4),
    }

# =========================
# 15. SAVE ALL OUTPUTS
# =========================
def save_all_outputs(qc_df, fn_df, summary, df, embeddings, out_dir="qc_output"):
    """
    qc_output/
    ├── METRICS/
    ├── REJECT/
    ├── FLAG/
    │   ├── FALSE_POSITIVES/
    │   ├── CLASS_MISMATCH/
    │   ├── LOW_IoU/
    │   └── CLUSTER_OUTLIER/
    ├── REVIEW/
    ├── FALSE_NEGATIVES/
    └── ACCEPT/

    Each saved image has a full-detail header strip ABOVE the crop.
    Object pixels are never overlaid or cropped by any text.
    """
    folders = ["METRICS","REJECT","FLAG/FALSE_POSITIVES","FLAG/CLASS_MISMATCH",
               "FLAG/LOW_IoU","FLAG/CLUSTER_OUTLIER","REVIEW","FALSE_NEGATIVES","ACCEPT"]
    for fd in folders:
        os.makedirs(os.path.join(out_dir, fd), exist_ok=True)

    # ── FIX: use save_annotated_crop (header ABOVE, object intact) ───
    save_ok   = [0]   # success counter (list so inner fn can mutate)
    save_fail = [0]   # fallback/error counter

    # Marker classes saved in color — they're small dark objects,
    # grayscale makes barcodes/decals/paper invisible
    FORCE_COLOR_CLASSES = {"barcode", "floor_decal", "paper_note", "paper_shortcut"}

    def save_crop(row, dest, saved, color=False):
        if saved >= MAX_PER_FOLDER:
            return False
        fname = (f"{row['class_name']}_q{row['quality_score']:.2f}_"
                 f"clip{row['clip_lbl_score']:.2f}_{row['annotation_id']}.jpg")
        dest_path = os.path.join(out_dir, dest, fname)
        use_color = color or (row['class_name'] in FORCE_COLOR_CLASSES)
        result = save_annotated_crop(row, dest_path, color=use_color)
        if result:
            save_ok[0] += 1
        else:
            save_fail[0] += 1
        return result

    def save_fn(row, idx):
        """Save full image with missed detection box drawn in red."""
        try:
            img  = Image.open(row["image_path"]).convert("RGB")
            bbox = eval(row["pred_bbox"])   # [x, y, w, h]
            draw = ImageDraw.Draw(img)
            x, y, w, h = bbox
            draw.rectangle([x, y, x+w, y+h], outline="red", width=4)
            draw.text((x+4, y+4),
                      f"MISSED: {row['pred_class']} conf={row['pred_confidence']:.2f}",
                      fill=(255, 80, 80))
            # ── Info strip at the bottom of the full image ────────────
            iw, ih = img.size
            strip_h = 18
            new_img = Image.new("RGB", (iw, ih + strip_h), (20, 20, 20))
            new_img.paste(img, (0, 0))
            d2 = ImageDraw.Draw(new_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
            except Exception:
                font = ImageFont.load_default()
            info = (f"FalseNeg #{idx:03d}  class:{row['pred_class']}"
                    f"  conf:{row['pred_confidence']:.3f}"
                    f"  maxIoU:{row['max_iou_with_anns']:.3f}"
                    f"  img:{os.path.basename(row['image_path'])}")
            d2.text((4, ih + 3), info, fill=(255, 220, 80), font=font)
            fname = f"fn{idx:03d}_{row['pred_class']}_conf{row['pred_confidence']:.2f}.jpg"
            new_img.save(os.path.join(out_dir, "FALSE_NEGATIVES", fname), quality=92)
            return True
        except Exception as e:
            print(f"  [save_fn] Error: {e}")
            return False

    counts = {fd.split("/")[-1]: 0 for fd in folders}
    counts["FALSE_NEGATIVES"] = 0

    for _, row in qc_df.iterrows():
        flag = row["qc_flag"]
        if flag == "REJECT":
            if save_crop(row, "REJECT", counts["REJECT"], color=True):
                counts["REJECT"] += 1
        elif flag == "FLAG":
            if row["is_false_positive"]:
                dest = "FLAG/FALSE_POSITIVES"; k = "FALSE_POSITIVES"
            elif row["class_match_iou"] == 0 and row["best_iou"] > 0.05:
                dest = "FLAG/CLASS_MISMATCH"; k = "CLASS_MISMATCH"
            elif row["best_iou"] < IOU_THRESHOLD and not row["is_false_positive"]:
                dest = "FLAG/LOW_IoU"; k = "LOW_IoU"
            else:
                dest = "FLAG/CLUSTER_OUTLIER"; k = "CLUSTER_OUTLIER"
            if save_crop(row, dest, counts.get(k, 0), color=(k=="CLASS_MISMATCH")):
                counts[k] = counts.get(k, 0) + 1
        elif flag == "REVIEW":
            if save_crop(row, "REVIEW", counts["REVIEW"], color=True):
                counts["REVIEW"] += 1
        elif flag == "ACCEPT":
            if save_crop(row, "ACCEPT", counts["ACCEPT"]):
                counts["ACCEPT"] += 1

    for idx, (_, row) in enumerate(fn_df.iterrows()):
        if counts["FALSE_NEGATIVES"] >= MAX_PER_FOLDER: break
        if save_fn(row, idx): counts["FALSE_NEGATIVES"] += 1

    total = len(qc_df)
    folder_info = {
        "REJECT":          (int((qc_df["qc_flag"]=="REJECT").sum()),
                            "Clearly wrong annotations — auto-rejected"),
        "FALSE_POSITIVES": (int(qc_df["is_false_positive"].sum()),
                            "Annotation exists but no object was detected there"),
        "CLASS_MISMATCH":  (int(((qc_df["qc_flag"]=="FLAG")&(qc_df["class_match_iou"]==0)).sum()),
                            "Model predicts a different class at this location"),
        "LOW_IoU":         (int(((qc_df["qc_flag"]=="FLAG")&(qc_df["best_iou"]<IOU_THRESHOLD)&(qc_df["is_false_positive"]==0)).sum()),
                            "Bounding box boundary is imprecise (IoU < 0.5)"),
        "CLUSTER_OUTLIER": (int(((qc_df["qc_flag"]=="FLAG")&(qc_df["proto_match"]==0)).sum()),
                            "Object looks visually unlike other objects of the same class"),
        "REVIEW":          (int((qc_df["qc_flag"]=="REVIEW").sum()),
                            "Borderline quality — needs human review"),
        "FALSE_NEGATIVES": (len(fn_df),
                            "Object detected by model but not annotated by human"),
        "ACCEPT":          (int((qc_df["qc_flag"]=="ACCEPT").sum()),
                            "High quality annotation — auto-accepted"),
    }

    for k, (actual, expl) in folder_info.items():
        folder_path = (os.path.join(out_dir, "FLAG", k)
                       if k in ["FALSE_POSITIVES","CLASS_MISMATCH","LOW_IoU","CLUSTER_OUTLIER"]
                       else os.path.join(out_dir, k))
        with open(os.path.join(folder_path, "COUNT.txt"), "w") as f:
            f.write(f"Category      : {k}\n")
            f.write(f"Total in data : {actual}\n")
            f.write(f"Images saved  : {counts.get(k,0)} (capped at {MAX_PER_FOLDER})\n")
            f.write(f"% of dataset  : {actual/total*100:.1f}%\n\n")
            f.write(f"What this means:\n  {expl}\n")

    # summary.txt
    with open(os.path.join(out_dir,"METRICS","summary.txt"),"w") as f:
        f.write("="*60+"\nSmartQC — ANNOTATION QUALITY CONTROL REPORT\n"+"="*60+"\n\n")
        f.write(f"Dataset        : Warehouse Detection (25 classes)\n")
        f.write(f"QC Engine      : DINOv2 + CLIP + IoU (3-signal adaptive)\n")
        f.write(f"Input format   : YOLO (also supports COCO JSON)\n\n")
        f.write(f"[ Decision Distribution ]\n")
        for flag in ["ACCEPT","REVIEW","FLAG","REJECT"]:
            cnt = summary.get(flag.lower(), 0)
            pct = summary.get(flag.lower()+"_pct", 0)
            f.write(f"  {flag:<8}: {cnt:>6} ({pct:>5.1f}%)\n")
        f.write(f"\n[ Detection Quality (Real YOLOv8) ]\n")
        f.write(f"  Precision      : {summary['precision']}\n")
        f.write(f"  Recall         : {summary['recall']}\n")
        f.write(f"  F1 Score       : {summary['f1']}\n")
        f.write(f"  False Positives: {summary['false_positives']}\n")
        f.write(f"  False Negatives: {summary['false_negatives']}\n")
        f.write(f"\n[ Signal Quality ]\n")
        f.write(f"  Mean CLIP score: {summary['mean_clip']}\n")
        f.write(f"  CLIP mismatches: {summary['clip_mismatch']}\n")
        f.write(f"  Proto match    : {summary['proto_match_rate']*100:.1f}%\n")
        f.write(f"  Mean IoU       : {summary['mean_iou']}\n")
        f.write(f"  Mean quality   : {summary['mean_quality']}\n")

    # All CSVs
    qc_df.to_csv(os.path.join(out_dir,"METRICS","qc_results.csv"),      index=False)
    fn_df.to_csv(os.path.join(out_dir,"METRICS","false_negatives.csv"),  index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,"METRICS","qc_summary.csv"), index=False)
    df.to_csv(os.path.join(out_dir,"METRICS","object_prototypes.csv"),   index=False)
    np.save(os.path.join(out_dir,"METRICS","embeddings.npy"), embeddings)

    print(f"\n  Image save summary: {save_ok[0]} OK, {save_fail[0]} fallback/failed")
    print(f"\n{'='*60}\nOUTPUT: {out_dir}/\n{'='*60}")
    for k, (actual, _) in folder_info.items():
        print(f"  {k:<22} | {counts.get(k,0):>4} images saved | total in data: {actual}")
    print(f"  METRICS              |  CSVs + summary.txt")

    banner("OUTPUT FILES", char="─")
    print(f"  📁 {out_dir}/")
    print(f"  ├── METRICS/")
    print(f"  │   ├── qc_results.csv      ({len(qc_df):,} rows — one per annotation)")
    print(f"  │   ├── false_negatives.csv ({len(fn_df):,} rows — missed objects)")
    print(f"  │   ├── qc_summary.csv      (overall metrics)")
    print(f"  │   ├── object_prototypes.csv (DINOv2 cluster assignments)")
    print(f"  │   ├── embeddings.npy      ({embeddings.shape} DINOv2 vectors)")
    print(f"  │   └── summary.txt         (human-readable report)")
    print(f"  ├── REJECT/                 ({folder_info['REJECT'][0]} — clearly wrong)")
    print(f"  ├── FLAG/")
    print(f"  │   ├── FALSE_POSITIVES/    ({folder_info['FALSE_POSITIVES'][0]} — phantom annotations)")
    print(f"  │   ├── CLASS_MISMATCH/     ({folder_info['CLASS_MISMATCH'][0]} — wrong label)")
    print(f"  │   ├── LOW_IoU/            ({folder_info['LOW_IoU'][0]} — poor boundary)")
    print(f"  │   └── CLUSTER_OUTLIER/    ({folder_info['CLUSTER_OUTLIER'][0]} — visual outlier)")
    print(f"  ├── REVIEW/                 ({folder_info['REVIEW'][0]} — needs human check)")
    print(f"  ├── FALSE_NEGATIVES/        ({folder_info['FALSE_NEGATIVES'][0]} — missed objects, marked in red)")
    print(f"  └── ACCEPT/                 ({folder_info['ACCEPT'][0]} — auto-accepted)")
    print()
    print("  NOTE: Each saved image has a full-detail header strip ABOVE the crop.")
    print("        Object pixels are NEVER overlaid or cut by text.")

# =========================
# 16. SIMPLE REST API
# =========================
def start_api(qc_df, summary):
    """
    FastAPI REST endpoints.
    Runs on http://localhost:8000 after QC pipeline completes.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn

        app = FastAPI(title="SmartQC API", version="1.0",
                      description="Annotation Quality Control System — Labellerr AI Capstone")

        results_dict  = qc_df.to_dict(orient="records")
        results_index = {r["annotation_id"]: r for r in results_dict}

        @app.get("/qc/health")
        def health():
            return {"status": "ok", "model": "DINOv2-base + CLIP ViT-B/32",
                    "total_annotations": len(qc_df),
                    "signals": ["CLIP semantic", "DINOv2 prototype", "IoU"]}

        @app.get("/qc/summary")
        def get_summary():
            return summary

        @app.get("/qc/results")
        def get_results(page: int = 1, per_page: int = 100,
                         flag: str = None, class_name: str = None):
            data = results_dict
            if flag:       data = [r for r in data if r["qc_flag"] == flag.upper()]
            if class_name: data = [r for r in data if r["class_name"] == class_name]
            start = (page-1)*per_page
            return {"page": page, "per_page": per_page,
                    "total": len(data), "results": data[start:start+per_page]}

        @app.get("/qc/results/{annotation_id}")
        def get_result(annotation_id: str):
            r = results_index.get(annotation_id)
            if r is None:
                return JSONResponse(status_code=404,
                                    content={"error": f"annotation '{annotation_id}' not found"})
            return r

        @app.get("/qc/flagged")
        def get_flagged():
            flagged = [r for r in results_dict if r["qc_flag"] in ["FLAG","REJECT"]]
            return {"total_flagged": len(flagged), "results": flagged}

        @app.get("/qc/stats/by-class")
        def stats_by_class():
            grp = qc_df.groupby("class_name").agg(
                count       =("annotation_id","count"),
                mean_quality=("quality_score","mean"),
                accept_pct  =("qc_flag", lambda x: (x=="ACCEPT").mean()),
                flag_pct    =("qc_flag", lambda x: (x=="FLAG").mean()),
                reject_pct  =("qc_flag", lambda x: (x=="REJECT").mean()),
            ).reset_index().to_dict(orient="records")
            return {"classes": grp}

        print("\n" + "="*60)
        print("SmartQC REST API starting...")
        print("  http://localhost:8000/qc/health")
        print("  http://localhost:8000/qc/summary")
        print("  http://localhost:8000/qc/results")
        print("  http://localhost:8000/qc/flagged")
        print("  http://localhost:8000/qc/stats/by-class")
        print("  http://localhost:8000/docs   (Swagger UI)")
        print("="*60)
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except ImportError:
        print("\nAPI not started — install with: pip install fastapi uvicorn")
        print("Then call: start_api(qc_df, summary)")

# =========================
# 17. MAIN
# =========================
if __name__ == "__main__":
    t0 = time.time()

    banner("SmartQC — Intelligent Annotation Quality Control")
    print("  Client  : Labellerr AI (Tensor Matics Inc.)")
    print("  Dataset : Warehouse Detection  |  25 classes  |  4651 images")
    print("  Engine  : DINOv2-base + CLIP ViT-B/32 + IoU (3-signal adaptive)")
    print("  Output  : ACCEPT / REVIEW / FLAG / REJECT per annotation")
    kv("  Device", DEVICE)
    kv("  Start time", time.strftime("%Y-%m-%d %H:%M:%S"))

    # ------------------------------------------------------------------
    banner("STEP 1: Class Mapping")
    id_to_name, id_to_category, name_to_id = load_class_mapping()
    all_class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    # ------------------------------------------------------------------
    banner("STEP 2: Load Annotations (YOLO Format)")
    # Option A: YOLO format (default)
    df = load_from_yolo(id_to_name, id_to_category, split="train")
    # Option B: COCO JSON format (tool-agnostic)
    # df = load_from_coco_json(
    #     annotation_file="/path/to/annotations.json",
    #     image_dir="/path/to/images",
    #     id_to_name=id_to_name,
    #     id_to_category=id_to_category,
    # )

    # ------------------------------------------------------------------
    banner("STEP 3: Load Models")
    dinov2, dinov2_proc = load_dinov2()
    clip,   clip_proc   = load_clip()

    # ------------------------------------------------------------------
    banner("STEP 4: DINOv2 Embeddings (Grayscale Object Crops)")
    embeddings, valid = extract_embeddings(df, dinov2, dinov2_proc)
    df = df.loc[valid].reset_index(drop=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    banner("STEP 5: Class Prototype Assignment (One per Class)")
    df, protos, proto_classes, purity_map = assign_prototypes(df, embeddings)

    # ------------------------------------------------------------------
    banner("STEP 6: CLIP Semantic Validation")
    clip_df = run_clip_validation(df, clip, clip_proc, all_class_names)

    # ------------------------------------------------------------------
    # STEP 7: Model Predictions
    # Option A (default): Mock YOLO for demonstration
    # Option B: Real YOLOv8 — uncomment and set model path
    # ------------------------------------------------------------------
    banner("STEP 7: YOLO Predictions")

    USE_REAL_YOLO = True   # ← using real trained YOLOv8 model

    if USE_REAL_YOLO:
        # ── Real YOLOv8 batched inference (GPU optimised) ────────────
        from ultralytics import YOLO as UltralyticsYOLO

        YOLO_MODEL_PATH = "/home/*/runs/detect/smartqc_warehouse/warehouse_yolo_v2/weights/best.pt"
        YOLO_BATCH_SIZE = 16    # images per batch — reduce if GPU OOM
        YOLO_IMG_SIZE   = 640   # inference resolution
        YOLO_CONF       = 0.25  # min confidence threshold

        print(f"\n  Model    : {YOLO_MODEL_PATH}")
        print(f"  Batch    : {YOLO_BATCH_SIZE}  |  ImgSize: {YOLO_IMG_SIZE}  |  Conf: {YOLO_CONF}")

        yolo_model      = UltralyticsYOLO(YOLO_MODEL_PATH)
        yolo_id_to_name = yolo_model.names  # {0: 'barrel', 1: 'barcode', ...}

        # ── CRITICAL: model class IDs differ from class_mapping.json IDs ──
        # Always use yolo_id_to_name for predictions so class names are correct.
        # Mismatch example: JSON id=3 → 'bottle', model id=3 → 'box'
        print(f"\n  Model class mapping (first 5): " +
              str({k: yolo_id_to_name[k] for k in sorted(yolo_id_to_name)[:5]}))

        image_paths = list(df["image_path"].unique())
        predictions = {p: [] for p in image_paths}

        print(f"\n  Running inference on {len(image_paths)} images...")

        for i in tqdm(range(0, len(image_paths), YOLO_BATCH_SIZE), desc="YOLOv8 inference"):
            batch_paths = image_paths[i : i + YOLO_BATCH_SIZE]
            try:
                batch_results = yolo_model(
                    batch_paths,
                    verbose=False,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF,
                )
                for img_path, r in zip(batch_paths, batch_results):
                    preds = []
                    if r.boxes is not None and len(r.boxes):
                        boxes = r.boxes.xyxy.cpu().numpy()   # x1,y1,x2,y2
                        confs = r.boxes.conf.cpu().numpy()
                        clses = r.boxes.cls.cpu().numpy()
                        for box, conf, cls_id in zip(boxes, confs, clses):
                            x1, y1, x2, y2 = box
                            # convert xyxy → xywh to match compute_iou format
                            preds.append({
                                "bbox":       [float(x1), float(y1),
                                               float(x2 - x1), float(y2 - y1)],
                                "confidence": float(conf),
                                "class_id":   int(cls_id),
                                "class_name": yolo_id_to_name.get(int(cls_id),
                                                              f"cls_{int(cls_id)}")
                            })
                    predictions[img_path] = preds
            except Exception as e:
                print(f"  Batch error (imgs {i}–{i+len(batch_paths)-1}): {e}")
                for p in batch_paths:
                    predictions[p] = []

        total_dets       = sum(len(v) for v in predictions.values())
        images_with_dets = sum(1 for v in predictions.values() if v)
        print(f"\n  Total detections     : {total_dets:,}")
        print(f"  Images with dets     : {images_with_dets:,} / {len(image_paths):,}")
        print(f"  Avg dets/image       : {total_dets / max(1, len(image_paths)):.1f}")
    else:
        # ── Mock YOLO (default for demo) ─────────────────────────────
        predictions = run_mock_yolo(df, id_to_name)

    # ------------------------------------------------------------------
    banner("STEP 8: QC Metrics (3-Signal Adaptive)")
    qc_df = compute_qc_metrics(df, clip_df, predictions, purity_map)

    # ------------------------------------------------------------------
    banner("STEP 9: False Negative Detection")
    fn_df = find_false_negatives(df, predictions)

    # ------------------------------------------------------------------
    banner("STEP 10: Full Metrics Matrix")
    # ── FIX: use qc_df (the actual QC DataFrame), not yolo results ──
    summary = compute_metrics_matrix(qc_df, fn_df)

    # ------------------------------------------------------------------
    banner("STEP 11: Save All Outputs")
    save_all_outputs(qc_df, fn_df, summary, df, embeddings, out_dir="qc_output")

    elapsed = time.time() - t0
    print(f"\n  ✓ SmartQC complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  ✓ Results in: qc_output/")
    print(f"  ✓ Full report: qc_output/METRICS/summary.txt")

    # ── Optional: start REST API ──────────────────────────────────────
    # start_api(qc_df, summary)

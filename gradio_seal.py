import os
import sys
import glob
import json
import argparse

import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), "event_encoder"))

from omegaconf import OmegaConf
from event_encoder.models.seal import SEAL
from event_encoder.data_utils.data_class import (
    DSEC19_CLASS, DSEC19_INSTANCE_ID, DSEC19_PRED_TO_ID,
    DSEC11_CLASS, DSEC11_INSTANCE_ID, DSEC11_PRED_TO_ID,
    DDD17_CLASS, DDD17_INSTANCE_ID, DDD17_PRED_TO_ID,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
seal_model = None
device = None
current_evimg_raw = None
current_voxel_path = None  # to derive instance mask path
clip_model = None          # lazy-loaded on first text query

classifiers = {}

DATASET_INFO = {
    "DSEC19": {
        "classes": DSEC19_CLASS,
        "instance_ids": DSEC19_INSTANCE_ID,
        "pred_to_id": DSEC19_PRED_TO_ID,
        "classifier_path": "classifier/DSEC19.pt",
    },
    "DSEC11": {
        "classes": DSEC11_CLASS,
        "instance_ids": DSEC11_INSTANCE_ID,
        "pred_to_id": DSEC11_PRED_TO_ID,
        "classifier_path": "classifier/DSEC11.pt",
    },
    "DDD17": {
        "classes": DDD17_CLASS,
        "instance_ids": DDD17_INSTANCE_ID,
        "pred_to_id": DDD17_PRED_TO_ID,
        "classifier_path": "classifier/DDD17.pt",
    },
}

IMG_SIZE = 512

# Distinct colours for top-k mask overlay
TOPK_COLORS = [
    (30, 144, 255),   # dodger blue
    (255, 80, 80),    # red
    (50, 205, 50),    # lime green
    (255, 165, 0),    # orange
    (180, 80, 255),   # purple
    (0, 220, 220),    # cyan
    (255, 200, 50),   # gold
    (255, 105, 180),  # hot pink
]

# ---------------------------------------------------------------------------
# JavaScript: single canvas overlay handles BOTH point clicks and box drags.
#
# - Box mode:  drag >= 5px  -> sends {type:"box", ...}
#              drag < 5px   -> ignored (tiny accidental drag)
# - Point mode: any click   -> sends {type:"point", ...}
#
# The mode is communicated from Python via a hidden textbox (#seal-mode)
# so there are no fragile DOM queries for the radio state.
# ---------------------------------------------------------------------------
DRAW_JS = """
() => {
    const IMG_SIZE = 512;
    let canvas = null;
    let drawing = false;
    let startX = 0, startY = 0;

    function getImg() {
        const c = document.getElementById('event-image');
        return c ? c.querySelector('img') : null;
    }

    function getMode() {
        const el = document.querySelector('#seal-mode textarea');
        if (el && el.value) return el.value.trim();
        return 'Box';
    }

    function send(obj) {
        obj._ts = Date.now();
        const el = document.querySelector('#prompt-data textarea');
        if (!el) return;
        const setter = Object.getOwnPropertyDescriptor(
            HTMLTextAreaElement.prototype, 'value').set;
        setter.call(el, JSON.stringify(obj));
        el.dispatchEvent(new Event('input', {bubbles: true}));
    }

    // Account for object-fit:contain letterboxing
    function imgTransform(img) {
        const nw = img.naturalWidth || IMG_SIZE;
        const nh = img.naturalHeight || IMG_SIZE;
        const dw = img.clientWidth, dh = img.clientHeight;
        const ratio = Math.min(dw / nw, dh / nh);
        const rw = nw * ratio, rh = nh * ratio;
        return {ox: (dw - rw) / 2, oy: (dh - rh) / 2, rw, rh};
    }

    function toImg(mx, my, t) {
        return [
            Math.round(Math.max(0, Math.min(IMG_SIZE - 1, (mx - t.ox) / t.rw * IMG_SIZE))),
            Math.round(Math.max(0, Math.min(IMG_SIZE - 1, (my - t.oy) / t.rh * IMG_SIZE)))
        ];
    }

    function setupCanvas() {
        const img = getImg();
        if (!img || !img.complete || img.naturalWidth === 0) return;

        const old = document.getElementById('draw-overlay');
        if (old) old.remove();

        canvas = document.createElement('canvas');
        canvas.id = 'draw-overlay';

        const imgRect = img.getBoundingClientRect();
        const parentRect = img.parentElement.getBoundingClientRect();
        canvas.width = Math.round(imgRect.width);
        canvas.height = Math.round(imgRect.height);

        Object.assign(canvas.style, {
            position: 'absolute',
            top: Math.round(imgRect.top - parentRect.top) + 'px',
            left: Math.round(imgRect.left - parentRect.left) + 'px',
            zIndex: '50',
            cursor: 'crosshair',
        });

        img.parentElement.style.position = 'relative';
        img.parentElement.appendChild(canvas);

        canvas.addEventListener('mousedown', (e) => {
            const mode = getMode();
            if (mode === 'Text') return;  // disable canvas in Text mode
            e.preventDefault();
            drawing = true;
            const r = canvas.getBoundingClientRect();
            startX = e.clientX - r.left;
            startY = e.clientY - r.top;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!drawing) return;
            e.preventDefault();
            const r = canvas.getBoundingClientRect();
            const cx = e.clientX - r.left, cy = e.clientY - r.top;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const mode = getMode();
            if (mode === 'Box') {
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 2;
                ctx.strokeRect(startX, startY, cx - startX, cy - startY);
            }
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!drawing) return;
            drawing = false; e.preventDefault();
            const r = canvas.getBoundingClientRect();
            const endX = e.clientX - r.left, endY = e.clientY - r.top;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const img = getImg();
            if (!img) return;
            const t = imgTransform(img);
            const mode = getMode();

            if (mode === 'Point') {
                const [ix, iy] = toImg(startX, startY, t);
                send({type: 'point', x: ix, y: iy});
            } else {
                const dx = Math.abs(endX - startX), dy = Math.abs(endY - startY);
                if (dx > 5 || dy > 5) {
                    const [ix1, iy1] = toImg(Math.min(startX, endX), Math.min(startY, endY), t);
                    const [ix2, iy2] = toImg(Math.max(startX, endX), Math.max(startY, endY), t);
                    send({type: 'box', x1: ix1, y1: iy1, x2: ix2, y2: iy2});
                }
            }
        });

        canvas.addEventListener('mouseleave', () => {
            if (drawing) {
                drawing = false;
                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
            }
        });
    }

    new MutationObserver(() => {
        const img = getImg();
        if (img && !document.getElementById('draw-overlay')) {
            if (img.complete) setupCanvas();
            else img.addEventListener('load', setupCanvas, {once: true});
        }
    }).observe(document.body, {childList: true, subtree: true});

    setInterval(() => {
        if (!document.getElementById('draw-overlay')) setupCanvas();
    }, 800);

    setTimeout(setupCanvas, 500);
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_preprocess_evimg(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(image_rgb)
    tensor = torch.nn.functional.interpolate(
        tensor[None], size=IMG_SIZE, mode="bilinear", align_corners=False
    )[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = (tensor - mean) / std
    display = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    return tensor, display


def load_display_image(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def get_rgb_path(voxel_path):
    return voxel_path.replace("/voxel_image/", "/rgb_image/")


def get_voxel_path(events_vis_path):
    scene_dir = os.path.dirname(os.path.dirname(events_vis_path))
    frame_name = os.path.basename(events_vis_path)
    return os.path.join(scene_dir, "voxel_image", frame_name)


def resize_mask_for_display(mask_np, shape_hw):
    h, w = shape_hw
    if mask_np.shape != (h, w):
        return cv2.resize(
            mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return mask_np


def scale_box_for_display(box, shape_hw):
    h, w = shape_hw
    sx = w / IMG_SIZE
    sy = h / IMG_SIZE
    return np.array([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy])


def make_overlay(raw_image, mask_np, label_text, score,
                 points=None, box=None, alpha=0.45):
    vis = raw_image.copy()
    h, w = vis.shape[:2]
    sx = w / IMG_SIZE
    sy = h / IMG_SIZE
    mask_vis = resize_mask_for_display(mask_np, (h, w))

    if box is not None:
        box_vis = scale_box_for_display(box, (h, w))
        x1, y1, x2, y2 = box_vis.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if points is not None:
        for (px, py) in points:
            ptx, pty = int(px * sx), int(py * sy)
            cv2.circle(vis, (ptx, pty), 6, (255, 0, 0), -1)
            cv2.circle(vis, (ptx, pty), 7, (255, 255, 255), 2)

    color = np.array([30, 144, 255], dtype=np.uint8)
    vis[mask_vis] = (
        vis[mask_vis].astype(np.float32) * (1 - alpha)
        + color.astype(np.float32) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

    text = f"{label_text} ({score:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    if contours:
        cx, cy = contours[0][0][0]
        cy = max(cy - 10, th + 5)
    else:
        cx, cy = 10, 30
    cv2.rectangle(vis, (cx, cy - th - 4), (cx + tw + 4, cy + 4), (0, 0, 0), -1)
    cv2.putText(vis, text, (cx + 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def run_prediction(points, point_labels, box, dataset_name):
    global seal_model, current_evimg_raw, classifiers

    info = DATASET_INFO[dataset_name]
    fusion = classifiers[dataset_name]

    with torch.no_grad():
        pred_masks, pred_labels, pred_scores, pred_idx = seal_model.predict(
            input_points=points,
            input_labels=point_labels,
            input_box=box,
            fusion=fusion,
        )

    if pred_idx == -1 or pred_masks is None:
        return current_evimg_raw.copy(), "No mask predicted. Try a different prompt."

    pred_label_idx = pred_labels[pred_idx].item()
    class_names = info["classes"]
    label_name = (class_names[pred_label_idx]
                  if pred_label_idx < len(class_names) else f"class_{pred_label_idx}")
    score = pred_scores[pred_idx].item()
    mask_np = pred_masks[pred_idx].cpu().numpy()

    overlay = make_overlay(current_evimg_raw, mask_np, label_name, score,
                           points=points, box=box)

    prompt_type = "box" if box is not None else "point"
    info_text = (f"Predicted class: {label_name}\n"
                 f"Confidence: {score:.4f}\n"
                 f"Prompt: {prompt_type}\n"
                 f"Mask pixels: {int(mask_np.sum())}")
    return overlay, info_text


# ---------------------------------------------------------------------------
# Text mode helpers
# ---------------------------------------------------------------------------

def get_clip_model():
    """Lazy-load CLIP model on first text query."""
    global clip_model
    if clip_model is None:
        import clip as clip_module
        print("Loading CLIP model (first text query)...")
        clip_model, _ = clip_module.load("ViT-L/14@336px", device=device)
        print("CLIP model loaded.")
    return clip_model


def get_text_features(text_list):
    """Encode text queries with CLIP."""
    import clip as clip_module
    model = get_clip_model()
    with torch.no_grad():
        tokens = clip_module.tokenize(text_list, truncate=True)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.float()


def load_instance_masks(voxel_path):
    """Load pre-computed instance masks from instance_gt/.
    Returns list of (binary_mask_512x512, class_id) or None."""
    mask_path = voxel_path.replace("/voxel_image/", "/instance_gt/").replace(".png", ".pt")
    if not os.path.exists(mask_path):
        return None

    # instance_gt/*.pt: tensor (N, 440, 640), values = class_id or -1
    masks = torch.load(mask_path, weights_only=False)
    results = []
    for i in range(masks.shape[0]):
        m = masks[i]
        cls_ids = m.unique()
        cls_ids = cls_ids[cls_ids != -1]
        if len(cls_ids) == 0:
            continue
        binary = (m != -1).float()
        binary = torch.nn.functional.interpolate(
            binary[None, None], size=IMG_SIZE, mode="nearest"
        )[0, 0].bool()
        results.append(binary)
    return results if results else None


DEFAULT_BG_QUERIES = [
    "a object in a scene.",
    "a things in a scene.",
    "a stuff in a scene.",
    "a texture in a scene.",
]


def on_text_predict(text_input, top_k, dataset_name):
    """Text mode: find the top-k instance masks most similar to the user's text."""
    global seal_model, current_evimg_raw, current_voxel_path

    top_k = int(top_k)

    if current_evimg_raw is None:
        return current_evimg_raw, "Load an image first."
    if not text_input or not text_input.strip():
        return current_evimg_raw, "Enter a text query."
    if current_voxel_path is None:
        return current_evimg_raw, "No sample loaded."

    # Build query list: user queries + default background for contrast
    queries = [q.strip() for q in text_input.split(",") if q.strip()]
    full_queries = queries + [q for q in DEFAULT_BG_QUERIES if q not in queries]
    text_feat = get_text_features(full_queries)  # [N_total_queries, dim]

    # Load all instance masks for current image
    instance_masks = load_instance_masks(current_voxel_path)
    if instance_masks is None:
        return current_evimg_raw, "No instance masks found for this image."

    # Collect all candidates that match the first query
    candidates = []

    for mask in instance_masks:
        box = seal_model.generate_prompt(mask.cpu().numpy(), type="box")
        if box is None:
            continue

        with torch.no_grad():
            masks_out, scores_out, tokens = seal_model.predict_mask(
                input_points=None,
                input_labels=None,
                input_box=box,
            )

        if len(masks_out) == 0:
            continue

        with torch.no_grad():
            labels, class_scores, costmap = seal_model.predict_label(
                masks_out, tokens, text_feat
            )

        total_scores = scores_out * class_scores
        pred_idx = total_scores.argmax().item()

        # Only keep masks where argmax label is 0 (target query)
        if labels[pred_idx].item() != 0:
            continue
        sim_score = costmap[pred_idx][0].item()
        candidates.append((
            masks_out[pred_idx].cpu().numpy(),
            queries[0],
            sim_score,
            box,
        ))

    if not candidates:
        return current_evimg_raw, "No valid masks found for this image."

    # Sort by similarity descending, take top-k
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:top_k]

    # Overlay all top-k masks with distinct colours
    vis = current_evimg_raw.copy()
    h, w = vis.shape[:2]
    info_lines = [f"Query: \"{queries[0]}\"",
                  f"Candidates searched: {len(instance_masks)}",
                  f"Matched: {len(candidates)}, showing top {len(top_candidates)}",
                  ""]

    alpha = 0.45
    for rank, (mask_np, label_name, score, box) in enumerate(top_candidates):
        color = np.array(TOPK_COLORS[rank % len(TOPK_COLORS)], dtype=np.uint8)
        mask_vis = resize_mask_for_display(mask_np, (h, w))
        box_vis = scale_box_for_display(box, (h, w))

        # Draw box
        x1, y1, x2, y2 = box_vis.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2),
                       tuple(int(c) for c in color), 2)

        # Overlay mask
        vis[mask_vis] = (
            vis[mask_vis].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)

        # Contour
        contours, _ = cv2.findContours(
            mask_vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

        # Label text
        text = f"#{rank+1} {label_name} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        if contours:
            cx, cy = contours[0][0][0]
            cy = max(cy - 10, th + 5)
        else:
            cx, cy = 10, 30 + rank * 25
        cv2.rectangle(vis, (cx, cy - th - 4), (cx + tw + 4, cy + 4),
                       tuple(int(c) for c in color), -1)
        cv2.putText(vis, text, (cx + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_lines.append(f"  #{rank+1}  similarity: {score:.4f}  pixels: {int(mask_np.sum())}")

    return vis, "\n".join(info_lines)


# ---------------------------------------------------------------------------
# Sample browser
# ---------------------------------------------------------------------------

def discover_samples(data_root):
    samples = {}
    vis_dirs = sorted(glob.glob(os.path.join(data_root, "*/events_vis")))
    for vis_dir in vis_dirs:
        scene = os.path.basename(os.path.dirname(vis_dir))
        files = sorted(glob.glob(os.path.join(vis_dir, "*.png")))
        if files:
            samples[scene] = files
    return samples


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_load_sample(scene, index, sample_dict, vis_source):
    global current_evimg_raw, current_voxel_path
    if scene not in sample_dict or not sample_dict[scene]:
        return None, None, [], "No samples found."

    idx = int(index) % len(sample_dict[scene])
    events_vis_path = sample_dict[scene][idx]
    voxel_path = get_voxel_path(events_vis_path)
    voxel_bgr = cv2.imread(voxel_path)
    if voxel_bgr is None:
        return None, None, [], f"Failed to read voxel frame: {voxel_path}"

    vis_bgr = cv2.imread(events_vis_path)
    if vis_bgr is None:
        return None, None, [], f"Failed to read events_vis frame: {events_vis_path}"

    evimg_tensor, _ = load_and_preprocess_evimg(voxel_bgr)
    if vis_source == "voxel":
        display = load_display_image(voxel_bgr)
    else:
        display = load_display_image(vis_bgr)
    seal_model.set_image(evimg_tensor)
    current_evimg_raw = display.copy()
    current_voxel_path = voxel_path

    rgb_path = get_rgb_path(voxel_path)
    rgb_display = None
    if os.path.exists(rgb_path):
        rgb_bgr = cv2.imread(rgb_path)
        if rgb_bgr is not None:
            rgb_display = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    status = (
        f"Loaded: {os.path.basename(events_vis_path)} from {scene} "
        f"(view={vis_source})"
    )
    return display, rgb_display, [], status


def on_prompt_received(data_str, points_state, dataset_name):
    """Single handler for both point and box, dispatched from JS."""
    global current_evimg_raw
    if not data_str or current_evimg_raw is None:
        return gr.update(), points_state, ""

    try:
        data = json.loads(data_str)
    except (json.JSONDecodeError, TypeError):
        return gr.update(), points_state, ""

    if data.get("type") == "point":
        points_state.append((data["x"], data["y"]))
        pts = np.array(points_state)
        lbls = np.ones(len(points_state))
        overlay, info_text = run_prediction(pts, lbls, None, dataset_name)
        return overlay, points_state, info_text

    elif data.get("type") == "box":
        box = np.array([data["x1"], data["y1"], data["x2"], data["y2"]])
        overlay, info_text = run_prediction(None, None, box, dataset_name)
        return overlay, [], info_text  # reset points on box draw

    return gr.update(), points_state, ""


def on_mode_change(mode):
    """When user switches mode, return the mode string so JS can read it
    from the hidden #seal-mode textbox, and toggle text input visibility.
    In Text mode, disable the dataset dropdown."""
    is_text = (mode == "Text")
    return (
        mode,
        gr.update(visible=is_text),
        gr.update(interactive=not is_text),
    )


def on_clear():
    global current_evimg_raw
    if current_evimg_raw is None:
        return None, [], ""
    return current_evimg_raw.copy(), [], "Cleared."


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_demo(sample_dict):
    scene_names = list(sample_dict.keys())

    with gr.Blocks(title="SEAL Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SEAL: Segment Any Events via Language")
        gr.Markdown(
            "Load an event image, then **drag** to draw a bounding box, "
            "switch to Point mode and **click** to place points, "
            "or use **Text** mode to find objects by language query."
        )

        # --- State ---
        sample_state = gr.State(sample_dict)
        click_state = gr.State([])

        # --- Hidden textboxes for JS <-> Python communication ---
        prompt_data = gr.Textbox(elem_id="prompt-data", visible=False)
        mode_box = gr.Textbox(value="Box", elem_id="seal-mode", visible=False)

        # --- Sample selection ---
        with gr.Row():
            scene_dropdown = gr.Dropdown(
                choices=scene_names,
                value=scene_names[0] if scene_names else None,
                label="Scene", scale=2,
            )
            first_max = len(sample_dict[scene_names[0]]) - 1 if scene_names else 0
            sample_slider = gr.Slider(
                minimum=0, maximum=first_max, step=1, value=0,
                label="Sample Index", scale=2,
            )
            vis_source = gr.Radio(
                choices=["event", "voxel"],
                value="event",
                label="Visualization",
                scale=1,
            )
            load_btn = gr.Button("Load Sample", variant="primary", scale=1)

        status_text = gr.Textbox(label="Status", interactive=False)

        def on_scene_change(scene, sd):
            n = len(sd.get(scene, [])) - 1
            return gr.update(maximum=max(n, 0), value=0)

        # --- Controls ---
        with gr.Row():
            prompt_mode = gr.Radio(
                choices=["Point", "Box", "Text"], value="Box",
                label="Prompt Mode", elem_id="prompt-mode", scale=1,
            )
            dataset_dropdown = gr.Dropdown(
                choices=list(DATASET_INFO.keys()), value="DSEC19",
                label="Class Template", scale=1,
            )
            clear_btn = gr.Button("Clear", scale=1)

        # --- Text mode input (hidden by default) ---
        with gr.Row(visible=False) as text_input_row:
            text_input = gr.Textbox(
                label="Text Query",
                placeholder="e.g. car, background",
                info="Enter target class first, then background classes separated by commas",
                scale=3,
            )
            topk_slider = gr.Slider(
                minimum=1, maximum=10, step=1, value=1,
                label="Top K", scale=1,
            )
            text_predict_btn = gr.Button("Predict", variant="primary", scale=1)

        # --- Main display ---
        with gr.Row():
            event_display = gr.Image(
                label="Event Image",
                type="numpy",
                interactive=False,
                show_download_button=False,
                elem_id="event-image",
            )
            rgb_display = gr.Image(
                label="RGB Reference",
                type="numpy",
                interactive=False,
                show_download_button=False,
            )

        # --- Prediction info ---
        result_text = gr.Textbox(
            label="Prediction Info", interactive=False, lines=4,
        )

        # --- Callbacks ---
        scene_dropdown.change(
            fn=on_scene_change,
            inputs=[scene_dropdown, sample_state],
            outputs=[sample_slider],
        )

        load_btn.click(
            fn=on_load_sample,
            inputs=[scene_dropdown, sample_slider, sample_state, vis_source],
            outputs=[event_display, rgb_display, click_state, status_text],
        )

        vis_source.change(
            fn=on_load_sample,
            inputs=[scene_dropdown, sample_slider, sample_state, vis_source],
            outputs=[event_display, rgb_display, click_state, status_text],
        )

        # JS sends prompt data (point or box) via hidden textbox
        prompt_data.input(
            fn=on_prompt_received,
            inputs=[prompt_data, click_state, dataset_dropdown],
            outputs=[event_display, click_state, result_text],
        )

        # When radio changes, write the mode to hidden textbox so JS can read it
        # and toggle text input row visibility
        prompt_mode.change(
            fn=on_mode_change,
            inputs=[prompt_mode],
            outputs=[mode_box, text_input_row, dataset_dropdown],
        )

        # Text mode predict button
        text_predict_btn.click(
            fn=on_text_predict,
            inputs=[text_input, topk_slider, dataset_dropdown],
            outputs=[event_display, result_text],
        )

        clear_btn.click(
            fn=on_clear, inputs=[],
            outputs=[event_display, click_state, result_text],
        )

        # Inject JS
        demo.load(fn=None, js=DRAW_JS)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SEAL Gradio Demo")
    parser.add_argument("--config", type=str, default="configs/eval/DSEC19/seal.yaml")
    parser.add_argument("--checkpoint", type=str, default="cache/seal.pt")
    parser.add_argument("--data-root", type=str, default="dataset/data/DSEC/test")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    global seal_model, device, classifiers

    args = parse_args()
    device = args.device
    cfg = OmegaConf.load(args.config)

    print("Loading SEAL model...")
    seal_model = SEAL(cfg.model).to(device)
    seal_model.eval()
    seal_model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    print("SEAL model loaded.")

    print("Loading text classifiers...")
    for name, info in DATASET_INFO.items():
        path = info["classifier_path"]
        if os.path.exists(path):
            classifiers[name] = torch.load(path, weights_only=True).to(device).float()
            print(f"  {name}: {path} ({classifiers[name].shape})")
        else:
            print(f"  {name}: {path} NOT FOUND (skipped)")
    print("Text classifiers loaded.")

    sample_dict = discover_samples(args.data_root)
    total = sum(len(v) for v in sample_dict.values())
    print(f"Found {total} samples across {len(sample_dict)} scenes.")

    demo = build_demo(sample_dict)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

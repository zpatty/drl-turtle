"""
Manual Stitch Minimal
=====================

A *bare‑bones*, **no automatic algorithm** interactive tool to help you manually
align two images (e.g., from two fixed cameras) by adjusting just a few
parameters:

    • Translation X (pixels)
    • Translation Y (pixels)
    • Rotation (degrees, about right image center before translation)
    • Uniform Scale
    • Blend Alpha (how strongly the right image is blended where overlap occurs)

You adjust sliders until the stitch looks good; then press **S** to save the
final parameters to a YAML file. Later, run in **batch mode** to apply the saved
transform automatically to many left/right image pairs.

**Deliberately simple:** No feature detection, no RANSAC, no auto seam, no heavy
GUI logic. Transparent NumPy + OpenCV code that you can easily hack.

---
## Quick Start
```bash
python manual_stitch_minimal.py left.jpg right.jpg --rig-name cam_pair
```
Adjust sliders. When satisfied:
- Press **s** → saves `cam_pair_params.yaml` next to the script (or use
  `--out-params path.yaml`).
- Press **q** or **ESC** to quit.

### Batch Apply Later
Prepare a CSV with lines: `left_path,right_path`.
```bash
python manual_stitch_minimal.py --batch pairs.csv --params cam_pair_params.yaml --outdir stitched_out
```

---
## Controls Summary
Window: *Stitch Preview*
- Trackbars: `dx`, `dy`, `rot`, `scale_x100`, `alpha`.
- Keyboard:
  - **s**: Save YAML params.
  - **w**: Write current stitched image to disk (uses `--debug-out` or current dir).
  - **r**: Reset params to defaults.
  - **q / ESC**: Quit.
  - **h**: Print help to console.

---
## Saved YAML Format
```yaml
model: similarity   # uniform scale + rotation + translation
scale: 1.000         # float
rot_deg: 0.0         # float, degrees (CCW)
tx: 0.0              # float, pixels (applied after rotation/scale about right image center)
ty: 0.0              # float, pixels
canvas_mode: auto    # 'auto' or 'fixed'
canvas_size: [W, H]  # used in batch if canvas_mode=='fixed'
alpha: 0.5           # blend weight for right image in overlap
```

You can edit this by hand.

---
## Implementation Notes
- Right image is first rotated & scaled about its own center, then translated by
  `(tx, ty)` **in the coordinate system of the left image** (top‑left = 0,0).
- The left image is pasted onto the canvas at (0,0) unchanged.
- Canvas size defaults to bounding box that fits both transformed right and
  original left images. Use `--fixed-canvas` to hold size constant across batch.
- Blend: simple alpha compositing where right image has nonzero pixels; alpha is
  the weight of the *right* image.

---
## Code Begins
"""

import os
import csv
import yaml
import argparse
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

# ----------------------------- Utility Functions -----------------------------

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def similarity_matrix(scale: float, rot_deg: float, tx: float, ty: float, cx: float, cy: float) -> np.ndarray:
    """Return 3x3 homography representing:
    1. Translate (-cx, -cy) to put right image center at origin.
    2. Scale.
    3. Rotate by rot_deg CCW.
    4. Translate back (+cx, +cy).
    5. Translate by (tx, ty) in *left image* pixel coords.
    """
    th = np.deg2rad(rot_deg)
    c, s = np.cos(th), np.sin(th)
    # Centering transforms
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    R = np.array([[ c, -s, 0], [ s,  c, 0], [0, 0, 1]], dtype=np.float32)
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
    T3 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    H = T3 @ T2 @ R @ S @ T1  # note right‑to‑left application
    return H.astype(np.float32)


def warp_right_into_left(img_left: np.ndarray,
                         img_right: np.ndarray,
                         scale: float,
                         rot_deg: float,
                         tx: float,
                         ty: float,
                         alpha: float,
                         fixed_canvas: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Apply manual similarity transform to right image and alpha blend with left.

    Returns stitched BGR image.
    """
    hL, wL = img_left.shape[:2]
    hR, wR = img_right.shape[:2]
    cx, cy = wR / 2.0, hR / 2.0

    H = similarity_matrix(scale, rot_deg, tx, ty, cx, cy)

    # Determine canvas size
    if fixed_canvas is not None:
        W, Hc = fixed_canvas
    else:
        # project right image corners
        corners = np.array([[0,0],[wR,0],[wR,hR],[0,hR]], dtype=np.float32).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, H)
        xs = warped[:,0,0]
        ys = warped[:,0,1]
        xs = np.concatenate([xs, np.array([0,wL], dtype=np.float32)])
        ys = np.concatenate([ys, np.array([0,hL], dtype=np.float32)])
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        # if translated negatively, shift canvas so left img stays at (0,0)
        shift_x = 0.0
        shift_y = 0.0
        if min_x < 0: shift_x = -min_x
        if min_y < 0: shift_y = -min_y
        W = int(np.ceil(max_x + shift_x))
        Hc = int(np.ceil(max_y + shift_y))
        # incorporate shift by adjusting global H and left placement
        if shift_x != 0 or shift_y != 0:
            Tshift = np.array([[1,0,shift_x],[0,1,shift_y],[0,0,1]], dtype=np.float32)
            H = Tshift @ H
            left_offset = (int(round(shift_x)), int(round(shift_y)))
        else:
            left_offset = (0,0)

    canvas = np.zeros((Hc, W, 3), dtype=np.uint8)

    # warp right
    warpedR = cv2.warpPerspective(img_right, H, (W, Hc))
    maskR = (cv2.warpPerspective(np.ones((hR,wR), dtype=np.uint8)*255, H, (W, Hc)) > 0)

    # paste left
    if fixed_canvas is not None:
        left_offset = (0,0)
    x0, y0 = left_offset
    canvas[y0:y0+hL, x0:x0+wL] = img_left
    maskL = np.zeros((Hc, W), dtype=bool)
    maskL[y0:y0+hL, x0:x0+wL] = True

    # alpha blend where both
    both = maskL & maskR
    onlyR = maskR & ~maskL
    onlyL = maskL & ~maskR

    out = canvas.copy()
    out[onlyR] = warpedR[onlyR]
    out[onlyL] = canvas[onlyL]  # already left
    if np.any(both):
        a = np.clip(alpha, 0.0, 1.0)
        out[both] = ( (1-a)*canvas[both].astype(np.float32) + a*warpedR[both].astype(np.float32) ).astype(np.uint8)

    return out


# ------------------------------- YAML I/O ------------------------------------

def save_params_yaml(path: str, params: Dict[str, Any]) -> None:
    with open(path, 'w') as f:
        yaml.safe_dump(params, f, sort_keys=False)


def load_params_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# -------------------------- Interactive UI State -----------------------------
class ManualStitchUI:
    def __init__(self, imgL, imgR, rig_name: str, out_params: Optional[str], fixed_canvas: Optional[Tuple[int,int]]):
        self.imgL = imgL
        self.imgR = imgR
        self.rig_name = rig_name
        self.out_params = out_params
        self.fixed_canvas = fixed_canvas
        # state
        self.dx = 0
        self.dy = 0
        self.rot = 0
        self.scale = 1.0
        self.alpha = 0.5
        self.window = "Stitch Preview"
        self._setup_ui()
        self.redraw()

    def _setup_ui(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('dx', self.window, 0, 2000, self._on_change)  # will shift -1000..+1000 by offset
        cv2.createTrackbar('dy', self.window, 0, 2000, self._on_change)
        cv2.createTrackbar('rot', self.window, 0, 360, self._on_change)
        cv2.createTrackbar('scale_x100', self.window, 100, 500, self._on_change)  # 0.00-5.00
        cv2.createTrackbar('alpha', self.window, 50, 100, self._on_change)  # 0-1 -> 0-100
        # set centers
        cv2.setTrackbarPos('dx', self.window, 1000)
        cv2.setTrackbarPos('dy', self.window, 1000)
        cv2.setTrackbarPos('rot', self.window, 180)

    def _read_trackbars(self):
        dx_raw = cv2.getTrackbarPos('dx', self.window)
        dy_raw = cv2.getTrackbarPos('dy', self.window)
        rot_raw = cv2.getTrackbarPos('rot', self.window)
        sc_raw  = cv2.getTrackbarPos('scale_x100', self.window)
        a_raw   = cv2.getTrackbarPos('alpha', self.window)
        self.dx = dx_raw - 1000
        self.dy = dy_raw - 1000
        self.rot = rot_raw - 180
        self.scale = max(sc_raw, 1) / 100.0
        self.alpha = a_raw / 100.0

    def _on_change(self, _=None):
        self._read_trackbars()
        self.redraw()

    def redraw(self):
        stitched = warp_right_into_left(self.imgL, self.imgR, self.scale, self.rot, self.dx, self.dy, self.alpha, self.fixed_canvas)
        cv2.imshow(self.window, stitched)
        self.current_preview = stitched

    def save_params(self):
        path = self.out_params if self.out_params else f"{self.rig_name}_params.yaml"
        params = dict(model='similarity', scale=float(self.scale), rot_deg=float(self.rot),
                      tx=float(self.dx), ty=float(self.dy), alpha=float(self.alpha),
                      canvas_mode=('fixed' if self.fixed_canvas else 'auto'),
                      canvas_size=list(self.fixed_canvas) if self.fixed_canvas else None)
        save_params_yaml(path, params)
        print(f"[INFO] Saved params -> {path}")

    def run(self):
        print_help_short()
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('s'):
                self.save_params()
            elif key == ord('r'):
                self._reset()
            elif key == ord('w'):
                out_name = f"{self.rig_name}_preview.png"
                cv2.imwrite(out_name, self.current_preview)
                print(f"[INFO] Wrote current preview -> {out_name}")
            elif key == ord('h'):
                print_help_short()
        cv2.destroyAllWindows()

    def _reset(self):
        cv2.setTrackbarPos('dx', self.window, 1000)
        cv2.setTrackbarPos('dy', self.window, 1000)
        cv2.setTrackbarPos('rot', self.window, 180)
        cv2.setTrackbarPos('scale_x100', self.window, 100)
        cv2.setTrackbarPos('alpha', self.window, 50)
        self._read_trackbars()
        self.redraw()


# ---------------------------- Batch Application ------------------------------

def apply_params_to_pair(left_path: str, right_path: str, params: Dict[str, Any]) -> np.ndarray:
    imgL = load_image(left_path)
    imgR = load_image(right_path)
    fixed = None
    if params.get('canvas_mode') == 'fixed' and params.get('canvas_size') is not None:
        fixed = tuple(int(v) for v in params['canvas_size'])
    out = warp_right_into_left(imgL, imgR,
                               scale=float(params['scale']),
                               rot_deg=float(params['rot_deg']),
                               tx=float(params['tx']),
                               ty=float(params['ty']),
                               alpha=float(params.get('alpha', 0.5)),
                               fixed_canvas=fixed)
    return out


def batch_apply(csv_path: str, params_path: str, outdir: str, prefix: str = "stitch_") -> None:
    params = load_params_yaml(params_path)
    os.makedirs(outdir, exist_ok=True)
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            if len(row) < 2:
                print(f"[WARN] skipping row (need left,right): {row}")
                continue
            left_path, right_path = row[0], row[1]
            try:
                stitched = apply_params_to_pair(left_path, right_path, params)
            except Exception as e:
                print(f"[ERROR] {left_path}, {right_path}: {e}")
                continue
            base = os.path.splitext(os.path.basename(left_path))[0]
            out_path = os.path.join(outdir, f"{prefix}{base}.png")
            cv2.imwrite(out_path, stitched)
            print(f"[OK] {out_path}")


# ------------------------------- CLI / Main ----------------------------------

def print_help_short():
    print("""
Manual Stitch Minimal Controls
-----------------------------
Sliders:
  dx, dy: translation in pixels (right image relative to left)
  rot: rotation degrees CCW
  scale_x100: uniform scale (value/100)
  alpha: blend weight for right image in overlap
Keys:
  s: save params YAML
  w: write current stitched preview PNG
  r: reset sliders
  h: print this help
  q or ESC: quit
""")


def parse_args():
    ap = argparse.ArgumentParser(description="Manual 2‑image stitching: no auto algorithm.")
    ap.add_argument('left', nargs='?', help='Left image path (interactive mode)')
    ap.add_argument('right', nargs='?', help='Right image path (interactive mode)')
    ap.add_argument('--rig-name', default='rig', help='Short name used when saving params/preview')
    ap.add_argument('--out-params', default=None, help='Path to save YAML params (interactive)')
    ap.add_argument('--params', default=None, help='Existing YAML params to load (pre‑set sliders or batch)')
    ap.add_argument('--batch', default=None, help='CSV file with left,right image paths (batch mode)')
    ap.add_argument('--outdir', default='stitch_batch_out', help='Output directory for batch mode')
    ap.add_argument('--fixed-canvas', default=None, type=str,
                    help='W,H to fix canvas across all images (e.g., 3840,1080). If omitted auto bbox is used.')
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse fixed canvas
    fixed_canvas = None
    if args.fixed_canvas:
        try:
            w,h = args.fixed_canvas.split(',')
            fixed_canvas = (int(w), int(h))
        except Exception:
            raise SystemExit("--fixed-canvas must be 'W,H'")

    # Batch mode
    if args.batch:
        if not args.params:
            raise SystemExit("Batch mode requires --params")
        batch_apply(args.batch, args.params, args.outdir)
        return

    # Interactive mode: need left & right
    if not args.left or not args.right:
        raise SystemExit("Interactive mode requires left and right image paths.")

    imgL = load_image(args.left)
    imgR = load_image(args.right)

    ui = ManualStitchUI(imgL, imgR, args.rig_name, args.out_params, fixed_canvas)

    # If params provided, preload sliders
    if args.params and os.path.isfile(args.params):
        try:
            params = load_params_yaml(args.params)
            # map values back to sliders
            ui.dx = int(round(float(params['tx']))) + 1000
            ui.dy = int(round(float(params['ty']))) + 1000
            ui.rot = int(round(float(params['rot_deg']))) + 180
            ui.scale = float(params['scale'])
            ui.alpha = float(params.get('alpha', 0.5))
            cv2.setTrackbarPos('dx', ui.window, ui.dx)
            cv2.setTrackbarPos('dy', ui.window, ui.dy)
            cv2.setTrackbarPos('rot', ui.window, ui.rot)
            cv2.setTrackbarPos('scale_x100', ui.window, max(int(round(ui.scale*100)),1))
            cv2.setTrackbarPos('alpha', ui.window, int(round(ui.alpha*100)))
            ui._read_trackbars()
            ui.redraw()
            print(f"[INFO] Loaded params from {args.params}")
        except Exception as e:
            print(f"[WARN] Could not load params {args.params}: {e}")

    ui.run()


if __name__ == '__main__':
    main()

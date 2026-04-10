import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Farnebäck Dense Optical Flow — pure NumPy + OpenCV (no scipy)
# ══════════════════════════════════════════════════════════════════════════════


# ── 1. Convolution helpers (no scipy) ────────────────────────────────────────


def _conv1d_rows(arr, kernel):
    """
    1-D convolution along axis=1 (each row) with reflect border padding.
    arr    : (H, W) float64
    kernel : 1-D float64 array of odd length
    """
    r = len(kernel) // 2
    padded = np.pad(arr, ((0, 0), (r, r)), mode="reflect")
    out = np.zeros_like(arr)
    for i, w in enumerate(kernel):
        out += w * padded[:, i : i + arr.shape[1]]
    return out


def _conv1d_cols(arr, kernel):
    """
    1-D convolution along axis=0 (each column) with reflect border padding.
    arr    : (H, W) float64
    kernel : 1-D float64 array of odd length
    """
    r = len(kernel) // 2
    padded = np.pad(arr, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(arr)
    for i, w in enumerate(kernel):
        out += w * padded[i : i + arr.shape[0], :]
    return out


# ── 2. Gaussian kernel ────────────────────────────────────────────────────────


def _gaussian_kernel_1d(sigma, truncate=3.0):
    """1-D Gaussian kernel normalised to sum = 1."""
    radius = int(sigma * truncate + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * x**2 / sigma**2)
    return k / k.sum()


# ── 3. Polynomial expansion ───────────────────────────────────────────────────


def poly_expand(gray, n=7, sigma=1.5):
    """
    Fit a 2-D quadratic polynomial to every pixel neighbourhood:
        f(x,y) ≈ r1·x² + r2·y² + r3·xy + r4·x + r5·y + r6

    Coefficients are solved via Gaussian-weighted least squares.
    The Gaussian applicability is separable, so the full 2-D fit reduces to
    products of 1-D convolutions — no per-pixel loop needed.

    Parameters
    ----------
    gray  : (H, W) uint8 or float grayscale image
    n     : half-window size  (full window = 2n+1 pixels)
    sigma : Gaussian sigma for the applicability (weight) kernel

    Returns  r1, r2, r3, r4, r5, r6  each (H, W) float64
    """
    img = gray.astype(np.float64)

    # Build weighted moment kernels
    app = _gaussian_kernel_1d(sigma, truncate=n / max(sigma, 1e-6))
    xs = np.arange(-len(app) // 2 + 1, len(app) // 2 + 1, dtype=np.float64)
    G_1 = app  # a(x)
    G_x = app * xs  # a(x)·x
    G_x2 = app * xs**2  # a(x)·x²

    # Moment images M_{pq} = Σ a(dx)·a(dy)·dx^p·dy^q·I(x+dx, y+dy)
    # Separability: first convolve rows with row-kernel, then cols with col-kernel
    def moment(row_k, col_k):
        return _conv1d_cols(_conv1d_rows(img, row_k), col_k)

    M00 = moment(G_1, G_1)  # plain Gaussian blur
    M10 = moment(G_x, G_1)  # x-weighted
    M01 = moment(G_1, G_x)  # y-weighted
    M20 = moment(G_x2, G_1)  # x²-weighted
    M02 = moment(G_1, G_x2)  # y²-weighted
    M11 = moment(G_x, G_x)  # xy-weighted

    # Normalisation scalars (constant for shift-invariant Gaussian)
    s0 = G_1.sum()  # Σ a(x)
    s2 = G_x2.sum()  # Σ a(x)·x²   (Σ a(x)·x = 0 by symmetry)

    N40 = s2 * s0
    N04 = s0 * s2
    N22 = s2 * s2
    N20 = s2 * s0
    N02 = s0 * s2
    N00 = s0 * s0

    # Solve three decoupled normal-equation blocks via Cramer's rule
    det_A = N40 * N04 - N22 * N22
    det_A = det_A if abs(det_A) > 1e-12 else 1e-12

    r1 = (N04 * M20 - N22 * M02) / det_A  # A[0,0]
    r2 = (-N22 * M20 + N40 * M02) / det_A  # A[1,1]
    r3 = M11 / (N22 + 1e-12)  # 2·A[0,1]
    r4 = M10 / (N20 + 1e-12)  # b[0]
    r5 = M01 / (N02 + 1e-12)  # b[1]
    r6 = M00 / (N00 + 1e-12)  # c

    return r1, r2, r3, r4, r5, r6


# ── 4. Single-level flow estimation ──────────────────────────────────────────


def _estimate_flow_single_level(
    r1_1,
    r2_1,
    r3_1,
    r4_1,
    r5_1,
    r1_2,
    r2_2,
    r3_2,
    r4_2,
    r5_2,
    flow_prior,
    smooth_sigma=1.5,
):
    """
    Solve the Farnebäck displacement equation per pixel:
        (A1 + A2) · d = −(b2 − b1) + (A1 + A2) · d_prior

    where  A = [[r1,   r3/2],    b = [r4, r5]
                [r3/2, r2  ]]

    Solved via Cramer's rule (no pixel loop).
    Flow is smoothed with a separable Gaussian afterwards.
    """
    M00 = r1_1 + r1_2  # (A1+A2)[0,0]
    M11 = r2_1 + r2_2  # (A1+A2)[1,1]
    M01 = (r3_1 + r3_2) * 0.5  # (A1+A2)[0,1]

    db0 = -(r4_2 - r4_1)
    db1 = -(r5_2 - r5_1)

    if flow_prior is not None:
        d0 = flow_prior[..., 0]
        d1 = flow_prior[..., 1]
        db0 += M00 * d0 + M01 * d1
        db1 += M01 * d0 + M11 * d1

    # Cramer's rule; regularise near-zero determinants (textureless regions)
    det = M00 * M11 - M01 * M01
    det = np.where(np.abs(det) < 1e-6, np.sign(det + 1e-12) * 1e-6, det)

    dx = (M11 * db0 - M01 * db1) / det
    dy = (-M01 * db0 + M00 * db1) / det

    flow = np.stack([dx, dy], axis=-1)

    # Post-solve Gaussian smoothing (pure NumPy)
    if smooth_sigma > 0:
        k = _gaussian_kernel_1d(smooth_sigma)
        flow[..., 0] = _conv1d_cols(_conv1d_rows(flow[..., 0], k), k)
        flow[..., 1] = _conv1d_cols(_conv1d_rows(flow[..., 1], k), k)

    return flow


# ── 5. Pyramidal Farnebäck ────────────────────────────────────────────────────


def build_pyramid(img, levels):
    """Gaussian image pyramid; index 0 = original, index -1 = coarsest."""
    pyr = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr


def pyramidal_farneback(
    im1, im2, levels=3, poly_n=7, poly_sigma=1.5, smooth_sigma=1.5, iterations=3
):
    """
    Coarse-to-fine dense optical flow via Farnebäck polynomial expansion.

    Parameters
    ----------
    im1, im2     : (H, W) uint8 grayscale frames
    levels       : pyramid depth
    poly_n       : half-window for polynomial expansion
    poly_sigma   : applicability kernel sigma
    smooth_sigma : Gaussian sigma for post-solve flow smoothing
    iterations   : refinement passes per pyramid level

    Returns
    -------
    flow : (H, W, 2) float32  —  (dx, dy) per pixel
    """
    pyr1 = build_pyramid(im1, levels)
    pyr2 = build_pyramid(im2, levels)

    flow = None

    for lvl in range(levels - 1, -1, -1):
        f1 = pyr1[lvl].astype(np.float64)
        f2 = pyr2[lvl].astype(np.float64)
        h, w = f1.shape

        # Upsample + scale flow from coarser level
        if flow is None:
            flow_up = np.zeros((h, w, 2), dtype=np.float64)
        else:
            flow_up = cv2.resize(flow, (w, h)) * 2.0

        r1_1, r2_1, r3_1, r4_1, r5_1, _ = poly_expand(f1, n=poly_n, sigma=poly_sigma)
        r1_2, r2_2, r3_2, r4_2, r5_2, _ = poly_expand(f2, n=poly_n, sigma=poly_sigma)

        flow = flow_up.copy()
        for _ in range(iterations):
            flow = _estimate_flow_single_level(
                r1_1,
                r2_1,
                r3_1,
                r4_1,
                r5_1,
                r1_2,
                r2_2,
                r3_2,
                r4_2,
                r5_2,
                flow_prior=flow,
                smooth_sigma=smooth_sigma,
            )

    return flow.astype(np.float32)


# ── 6. Visualization helpers ──────────────────────────────────────────────────


def flow_to_hsv(flow, max_magnitude=None):
    """Hue = direction, Value = speed, Saturation = 255."""
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = (angle / 2).astype(np.uint8)

    if max_magnitude is None:
        max_magnitude = float(magnitude.max()) or 1.0
    hsv[..., 2] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_to_arrows(flow, step=16, scale=3.0, threshold=0.5):
    """Sample a quiver grid from the flow field."""
    h, w = flow.shape[:2]
    arrows = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            if np.hypot(dx, dy) < threshold:
                continue
            x1 = int(round(x + dx * scale))
            y1 = int(round(y + dy * scale))
            arrows.append(((x, y), (x1, y1)))
    return arrows


def draw_arrows(frame, arrows, color=(0, 255, 0), thickness=1):
    out = frame.copy()
    for (x0, y0), (x1, y1) in arrows:
        cv2.arrowedLine(out, (x0, y0), (x1, y1), color, thickness, tipLength=0.3)
    return out


# ── Config ────────────────────────────────────────────────────────────────────

SCALE = 0.33
FRAME_SKIP = 2

LEVELS = 3
POLY_N = 7
POLY_SIGMA = 1.5
SMOOTH_SIGMA = 1.5
ITERATIONS = 3

VIZ_MODE = "blend"  # "hsv" | "arrows" | "blend"
ARROW_STEP = 16
ARROW_SCALE = 4.0
ARROW_THRESH = 0.8
MAX_MAG = None  # None = auto; set e.g. 15.0 for stable colours

# ── Setup ─────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture("OF_trimmed.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video file.")
    print("Make sure OF_trimmed.mp4 is in the same folder as this script.")
    exit()

print(
    f"Video opened.  FPS: {cap.get(cv2.CAP_PROP_FPS)},  "
    f"Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)},  "
    f"Size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}×{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
)

ret, old_frame = cap.read()
if not ret:
    print("ERROR: Could not read first frame.")
    exit()

old_frame = cv2.resize(old_frame, (0, 0), fx=SCALE, fy=SCALE)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
frame_count = 0

print(f"Starting dense optical flow (mode={VIZ_MODE}). Press Q to quit.")

cv2.namedWindow("Dense Optical Flow", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dense Optical Flow", 1280, 720)

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = pyramidal_farneback(
        old_gray,
        frame_gray,
        levels=LEVELS,
        poly_n=POLY_N,
        poly_sigma=POLY_SIGMA,
        smooth_sigma=SMOOTH_SIGMA,
        iterations=ITERATIONS,
    )

    if VIZ_MODE == "hsv":
        output = flow_to_hsv(flow, max_magnitude=MAX_MAG)

    elif VIZ_MODE == "arrows":
        arrows = flow_to_arrows(
            flow, step=ARROW_STEP, scale=ARROW_SCALE, threshold=ARROW_THRESH
        )
        output = draw_arrows(frame, arrows)

    elif VIZ_MODE == "blend":
        hsv_vis = flow_to_hsv(flow, max_magnitude=MAX_MAG)
        arrows = flow_to_arrows(
            flow, step=ARROW_STEP, scale=ARROW_SCALE, threshold=ARROW_THRESH
        )
        output = cv2.addWeighted(draw_arrows(frame, arrows), 0.6, hsv_vis, 0.4, 0)

    else:
        raise ValueError(f"Unknown VIZ_MODE: {VIZ_MODE!r}")

    cv2.imshow("Dense Optical Flow", output)
    old_gray = frame_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit by user.")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

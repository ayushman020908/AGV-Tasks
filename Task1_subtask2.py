"""
controller.py

Artificial Potential Field (APF) controller for PyBullet slalom simulation.
Uses simulation_setup.py for scene + car creation.

Sensing
-------
  PyBullet camera → grayscale → Otsu thresholding (from scratch)
  → Gaussian blur (from scratch) → gradient → obstacle repulsive force

APF components
--------------
  F_att   : goal attraction        U_att = ½α‖p − p_goal‖²
  F_obs   : obstacle repulsion     derived from ∇(G * O(x,y))
  F_road  : road boundary repulsion  Morse potential, bounded to ±ROAD_SAFE

Gap clearance
-------------
  GapAnalyser projects the Otsu binary mask onto a 1D lateral profile,
  runs a run-length pass to locate clear-column runs between obstacle blobs
  and the road walls, converts each gap from pixel columns into estimated
  world-space width (metres), and compares it against the car body width
  plus a configurable safety margin.

  Passable gap    → no extra modulation (gap_weight = 1.0)
  Impassable gap  → force toward that side is amplified by GAP_BLOCK_SCALE
                    so the APF steers the car away from blocked corridors

Control
-------
  Desired heading ψ_d = atan2(F_total_y, F_total_x)
  Lateral sliding mode on heading error ψ_e = ψ − ψ_d  →  steering angle δ
  Longitudinal sliding mode on speed error → wheel velocity

Usage
-----
    python controller.py
"""

import time
import math
import cv2
import numpy as np
import pybullet as p
from simulation_setup import setup_simulation


# =============================================================================
# PARAMETERS  — all tunable constants in one place
# =============================================================================

# ── Road geometry ─────────────────────────────────────────────────────────────
ROAD_HALF = 1.16  # physical half-width (m) — actual wall position
CAR_HALF_WIDTH = 0.36  # half of CAR_BODY_WIDTH (0.72/2)
# Car centre y at which the car EDGE touches the wall:
WALL_CONTACT_Y = ROAD_HALF - CAR_HALF_WIDTH  # 1.16 − 0.36 = 0.80 m

# Repulsion dead-zone: zero for |y| ≤ ROAD_SAFE, quadratic up to WALL_CONTACT_Y
# Decreased by 0.20 from previous 0.70 → kicks in earlier
ROAD_SAFE = 0.50  # repulsion is exactly 0 for |y| ≤ 0.50 m
# Buffer = WALL_CONTACT_Y − ROAD_SAFE = 0.80 − 0.50 = 0.30 m

# ── Goal ──────────────────────────────────────────────────────────────────────
GOAL_X = 30.0  # x position to stop at (m)
GOAL_Y = 0.0  # desired lateral position (m)

# ── APF gains ─────────────────────────────────────────────────────────────────
APF_ALPHA = 1.0  # attractive force magnitude (unit vector × α)
APF_A = 9.0  # raised 6→9: stronger wall push when car drifts close
APF_B = 1.0  # unused (kept for reference)
APF_GAMMA = 5.0  # obstacle repulsion gain γ

# ── Lane-centre restoration ────────────────────────────────────────────────────
# F_lane = −LANE_K · clamp(car_y, ±LANE_CLAMP)
# ONLY activates when:
#   (a) road_clear = True  (no obstacle pixels in camera)
#   (b) car has physically passed the last cone (car_x > cone_x + CAR_HALF_LENGTH)
#   (c) steering_bias == 0
LANE_K = 1.5  # stronger spring: pulls faster toward y=0
LANE_CLAMP = 0.50  # wider clamp: full spring even at larger offsets
LANE_DEADZONE = 0.08  # spring = 0 when |car_y| < this (no centre wiggle)

# ── Car geometry (for cone-clearance check) ────────────────────────────────────
# Cone positions come from simulation_setup — listed here for the lockout logic.
CAR_HALF_LENGTH = 0.35
CONE_X_POSITIONS = [6.0, 12.0, 18.0, 24.0]
CONE_Y_POSITIONS = [+0.38, -0.38, +0.38, -0.38]
CONE_HALF_WIDTH = 0.15  # half the physical cone width (m)

# Bias cancels once car centre is this far from cone centre in avoidance direction.
# = CAR_HALF_WIDTH + CONE_HALF_WIDTH + safety = 0.36 + 0.15 + 0.10 = 0.61 m
# Prevents the corridor bias from pushing the car all the way to the wall.
BIAS_CLEAR_MARGIN = 0.55  # m — lateral separation at which bias is zeroed

# ── Obstacle detection ────────────────────────────────────────────────────────
OBS_CROP_FRAC = 1 / 3  # fraction of image rows to discard from top (sky)
OBS_SIGMA_DIV = 6.0  # Gaussian σ = image_width / OBS_SIGMA_DIV

# Minimum obstacle pixel fraction before ANY repulsion force activates.
# Below this the obstacle is too far away / too small to matter.
# Eliminates wiggle from tiny far-away blobs and road-texture noise.
#   prox_weight = obstacle_pixels / road_crop_pixels
#   Typical values: far cone ≈ 0.03, close cone ≈ 0.15–0.25
OBS_MIN_PROX = 0.07  # raised from 0.04: avoids false-positive at start
OBS_SLOW_PROX = 0.12  # ≥12% → slow down
OBS_SLOW_SPEED = 4.0

# End-wall gate: disable obstacle detection in the final GOAL_CLEAR_DIST metres.
# The end wall fills the camera and causes spurious OBS state for the last stretch.
GOAL_CLEAR_DIST = 4.0  # metres before GOAL_X to stop obstacle detection

# Full-frame Otsu with polarity check
# ─────────────────────────────────────
# We run Otsu on the FULL frame (sky + road + cone).  The sky is the brightest
# region and pulls the histogram into a clear bimodal distribution:
#   class 0 (below thresh) = road surface + dark cone stripes
#   class 1 (above thresh) = sky + bright cone stripes
# After thresholding we keep only the road crop rows.
# We then check which polarity actually captures the cone:
#   The bottom OBS_FLOOR_FRAC rows of the crop are pure road.
#   If their mean in the binary image is HIGH (>0.5), the road surface
#   is being called "foreground" — we invert the mask.
# Finally, small isolated blobs from road markings are suppressed with
# morphological erosion (OBS_ERODE_KSIZE pixels).
OBS_FLOOR_FRAC = 0.20  # bottom fraction of crop rows = guaranteed road
OBS_ERODE_KSIZE = 3  # erosion kernel size to kill 1-2 px road noise (odd int)

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_IMG_W = 160
CAM_IMG_H = 120
CAM_FOV = 70.0
CAM_NEAR = 0.15
CAM_FAR = 15.0
CAM_OFFSET = [0.3, 0.0, 0.35]  # moved back from 0.75: on hood, slightly higher

# ── Gap clearance ─────────────────────────────────────────────────────────────
#
# Physical car body width (metres).  Adjust to match simulation_setup.py.
CAR_BODY_WIDTH = 0.72  # m  — lateral extent of the car chassis

# Minimum margin to add on each side of the car when testing a gap.
# gap is "passable" only when gap_world_width ≥ CAR_BODY_WIDTH + 2*GAP_MARGIN
GAP_MARGIN = 0.15  # m  — per side (total clearance = 2 × 0.15 = 0.30 m)

# How aggressively to repel toward a blocked gap.
# IMPORTANT: steering_bias is added directly to total_y while F_att_x ≈ 1.0.
# atan2(bias, 1.0) = the heading deflection it causes.
# atan2(0.4, 1.0) ≈ 22°  ← gentle but clear preference
# atan2(4.0, 1.0) ≈ 76°  ← nearly perpendicular = car spins out (old value)
GAP_BLOCK_SCALE = 0.4

# 3.0 m trigger: at v≈1.8 m/s → 1.7 s approach time.
# Spring window = 6.0 - 0.35 - 3.0 = 2.65 m between cones — enough to re-centre.
GAP_TRIGGER_DIST = 3.0  # metres before next cone: start corridor steering

# Minimum fraction of image columns a "clear run" must span to count as a gap.
# Filters single-pixel noise columns from being declared gaps.
GAP_MIN_COL_FRAC = 0.05  # 5 % of image width

# Horizontal FOV used for the pixel→world column mapping.
# Should match the camera's aspect ratio × CAM_FOV (approx).
GAP_HFOV_DEG = 90.0  # degrees (horizontal field of view estimate)

# Distance ahead at which the obstacle is assumed to sit for width estimation.
# This is the projection plane distance used in the pinhole column→world calc.
GAP_PROJ_DIST = 3.0  # m  — nominal obstacle distance for gap width calc

# ── Lucas-Kanade ──────────────────────────────────────────────────────────────
LK_LEVELS = 3
LK_WINDOW = 15
LK_MAX_ITER = 20
LK_EPSILON = 0.01

# ── Shi-Tomasi ────────────────────────────────────────────────────────────────
ST_MAX_CORNERS = 300
ST_QUALITY = 0.03
ST_MIN_DIST = 5
ST_BLOCK_SIZE = 7

# ── Debug visualizer ──────────────────────────────────────────────────────────
VIZ_SCALE = 3
VIZ_REDETECT_EVERY = 15

# ── Lateral sliding mode controller ───────────────────────────────────────────
SMC_CR = 3.0  # raised 2.0→3.0: faster heading convergence
SMC_U0 = 3.0  # raised 1.5→3.0: faster steering rate
MAX_STEER = 0.5
# Hard limit on desired heading angle from APF.
# Prevents any single large force term from commanding a near-perpendicular
# heading that spins the car out before the SMC can react.
# atan2(tan(30°), 1) = 30°  →  total_y is limited to tan(30°) × total_x
MAX_PSI_D = 0.5236  # base max desired heading = ±30° (π/6)
MAX_PSI_D_WALL = 1.0472  # max heading near wall = ±60° (π/3)
# Interpolated linearly between ROAD_SAFE and WALL_CONTACT_Y.

# Wall override: when |car_y| > WALL_OVERRIDE_Y, ALL other lateral forces are
# zeroed and ONLY the wall repulsion drives total_y.  This guarantees the car
# escapes the wall regardless of what the obstacle forces are doing.
WALL_OVERRIDE_Y = 0.60  # wall override triggers here — tighter than before

# ── Longitudinal sliding mode controller ──────────────────────────────────────
SPD_CL = 1.0
SPD_A0 = 5.0
TARGET_SPEED = 5.0  # reduced 8.0→5.0: more time to react per metre
MAX_WHEEL_SPEED = 20.0

# ── Drive ─────────────────────────────────────────────────────────────────────
WHEEL_FORCE = 800
SIM_DT = 1 / 60


# =============================================================================
# CAMERA HELPER
# =============================================================================


class CameraHelper:
    """Attaches a virtual camera to the car and returns greyscale frames."""

    def __init__(
        self,
        car_id,
        img_w=CAM_IMG_W,
        img_h=CAM_IMG_H,
        fov=CAM_FOV,
        near=CAM_NEAR,
        far=CAM_FAR,
        offset=None,
    ):
        self.car_id = car_id
        self.img_w = img_w
        self.img_h = img_h
        self.offset = offset if offset is not None else CAM_OFFSET
        self.proj = p.computeProjectionMatrixFOV(
            fov=fov, aspect=img_w / img_h, nearVal=near, farVal=far
        )

    def _view_matrix(self):
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_pos = np.array(pos) + R @ np.array(self.offset)
        target = np.array(pos) + R @ np.array([5.0, 0.0, 0.0])
        up = R @ np.array([0.0, 0.0, 1.0])
        return p.computeViewMatrix(cam_pos.tolist(), target.tolist(), up.tolist())

    def grab_gray(self) -> np.ndarray:
        """Returns (H, W) uint8 greyscale image from the car's front camera."""
        _, _, rgba, _, _ = p.getCameraImage(
            self.img_w,
            self.img_h,
            viewMatrix=self._view_matrix(),
            projectionMatrix=self.proj,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(self.img_h, self.img_w, 4)[
            :, :, :3
        ]
        gray = (
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        ).astype(np.uint8)
        return gray


# =============================================================================
# LUCAS-KANADE OPTICAL FLOW  (from scratch)
# =============================================================================


def build_pyramid(img, levels=5):
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def lk_single_level(im1, im2, p0, window_size=15, max_iter=20, epsilon=0.01):
    half_w = window_size // 2
    h, w = im1.shape
    im1f = im1.astype(np.float64)
    im2f = im2.astype(np.float64)
    Ix = cv2.Sobel(im1f, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(im1f, cv2.CV_64F, 0, 1, ksize=3)
    p1 = p0.copy().astype(np.float32)
    status = np.ones(len(p0), dtype=np.uint8)

    for i, pt in enumerate(p0):
        x0, y0 = pt[0, 0], pt[0, 1]
        xi, yi = int(round(x0)), int(round(y0))
        if xi - half_w < 0 or xi + half_w >= w or yi - half_w < 0 or yi + half_w >= h:
            status[i] = 0
            continue
        Ix_win = Ix[
            yi - half_w : yi + half_w + 1, xi - half_w : xi + half_w + 1
        ].flatten()
        Iy_win = Iy[
            yi - half_w : yi + half_w + 1, xi - half_w : xi + half_w + 1
        ].flatten()
        A = np.vstack([Ix_win, Iy_win]).T
        ATA = A.T @ A
        if np.linalg.eigvalsh(ATA)[0] < 1e-3:
            status[i] = 0
            continue
        ATA_inv = np.linalg.inv(ATA)
        vx, vy = 0.0, 0.0
        for _ in range(max_iter):
            nx, ny = x0 + vx, y0 + vy
            nxi, nyi = int(round(nx)), int(round(ny))
            if (
                nxi - half_w < 0
                or nxi + half_w >= w
                or nyi - half_w < 0
                or nyi + half_w >= h
            ):
                status[i] = 0
                break
            patch1 = im1f[yi - half_w : yi + half_w + 1, xi - half_w : xi + half_w + 1]
            patch2 = im2f[
                nyi - half_w : nyi + half_w + 1, nxi - half_w : nxi + half_w + 1
            ]
            It_win = (patch2 - patch1).flatten()
            delta = ATA_inv @ (-A.T @ It_win)
            vx += delta[0]
            vy += delta[1]
            if np.linalg.norm(delta) < epsilon:
                break
        if status[i]:
            p1[i, 0, 0] = x0 + vx
            p1[i, 0, 1] = y0 + vy
    return p1, status


def pyramidal_lucas_kanade(
    im1, im2, p0, levels=3, window_size=15, max_iter=20, epsilon=0.01
):
    pyr1 = build_pyramid(im1, levels)
    pyr2 = build_pyramid(im2, levels)
    flow = np.zeros_like(p0, dtype=np.float32)
    status = np.ones(len(p0), dtype=np.uint8)

    for lvl in range(levels - 1, -1, -1):
        scale = 2**lvl
        p_lvl = p0.astype(np.float32) / scale
        p_guess = p_lvl + flow
        p1_lvl, st = lk_single_level(
            pyr1[lvl],
            pyr2[lvl],
            p_guess,
            window_size=window_size,
            max_iter=max_iter,
            epsilon=epsilon,
        )
        flow = flow + (p1_lvl - p_guess)
        status &= st
        if lvl > 0:
            flow *= 2

    return p0.astype(np.float32) + flow, status


# =============================================================================
# DEBUG VISUALIZER
# =============================================================================


class DebugVisualizer:
    """Grayscale camera feed + LK flow vectors + telemetry overlay."""

    def __init__(
        self,
        window_name="Debug | Camera + LK Flow",
        scale=VIZ_SCALE,
        max_corners=ST_MAX_CORNERS,
        redetect_every=VIZ_REDETECT_EVERY,
    ):
        self.window_name = window_name
        self.scale = scale
        self.max_corners = max_corners
        self.redetect_every = redetect_every
        self._prev_gray = None
        self._prev_pts = None
        self._step = 0
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def _detect_corners(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=ST_MAX_CORNERS,
            qualityLevel=ST_QUALITY,
            minDistance=ST_MIN_DIST,
            blockSize=ST_BLOCK_SIZE,
        )

    def update(self, gray: np.ndarray, telemetry: dict) -> bool:
        H, W = gray.shape
        if (
            self._prev_pts is None
            or len(self._prev_pts) < 10
            or self._step % self.redetect_every == 0
        ):
            self._prev_pts = self._detect_corners(gray)

        flow_pairs = []
        if (
            self._prev_gray is not None
            and self._prev_pts is not None
            and len(self._prev_pts) > 0
        ):
            next_pts, status = pyramidal_lucas_kanade(
                self._prev_gray,
                gray,
                self._prev_pts,
                levels=LK_LEVELS,
                window_size=LK_WINDOW,
                max_iter=LK_MAX_ITER,
                epsilon=LK_EPSILON,
            )
            good_new = next_pts[status == 1]
            good_old = self._prev_pts[status == 1]
            for new, old in zip(good_new, good_old):
                flow_pairs.append((old[0], new[0]))
            self._prev_pts = good_new.reshape(-1, 1, 2)

        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x0, y0), (x1, y1) in flow_pairs:
            cv2.arrowedLine(
                display,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                (255, 200, 0),
                1,
                tipLength=0.4,
            )
        if self._prev_pts is not None:
            for pt in self._prev_pts:
                cv2.circle(display, (int(pt[0, 0]), int(pt[0, 1])), 2, (0, 255, 80), -1)

        # Draw gap passability overlay on the road region
        gaps = telemetry.get("gaps", [])
        crop_start = int(H * OBS_CROP_FRAC)
        for gap in gaps:
            col_l = gap["col_left"]
            col_r = gap["col_right"]
            passable = gap["passable"]
            color = (0, 220, 80) if passable else (0, 60, 220)
            # Draw vertical bracket lines in the cropped region
            cv2.line(display, (col_l, crop_start), (col_l, H - 1), color, 1)
            cv2.line(display, (col_r, crop_start), (col_r, H - 1), color, 1)
            mid_col = (col_l + col_r) // 2
            label_y = crop_start + 8
            label = "OK" if passable else "X"
            cv2.putText(
                display,
                label,
                (mid_col - 6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )

        display = cv2.resize(
            display, (W * self.scale, H * self.scale), interpolation=cv2.INTER_NEAREST
        )

        panel_h = 140
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (W * self.scale, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small = 0.42
        font_med = 0.50
        col1, col2 = 8, W * self.scale // 2 + 8
        green = (80, 220, 80)
        cyan = (255, 220, 0)
        yellow = (0, 220, 220)
        white = (220, 220, 220)
        red = (80, 80, 220)

        def put(text, x, y, color=white, scale=font_small):
            cv2.putText(display, text, (x, y), font, scale, color, 1, cv2.LINE_AA)

        put(f"Speed   : {telemetry.get('speed', 0):.2f} m/s", col1, 18, green, font_med)
        put(
            f"Yaw     : {telemetry.get('yaw_deg', 0):+.1f} deg",
            col2,
            18,
            green,
            font_med,
        )
        put(
            f"Steer   : {telemetry.get('steer_deg', 0):+.1f} deg",
            col1,
            40,
            cyan,
            font_med,
        )
        put(
            f"psi_d   : {telemetry.get('psi_d_deg', 0):+.1f} deg",
            col2,
            40,
            cyan,
            font_med,
        )
        put(
            f"Accel   : {telemetry.get('accel', 0):+.3f} m/s2",
            col1,
            62,
            yellow,
            font_med,
        )
        put(
            f"WheelSpd: {telemetry.get('wheel_speed', 0):.1f} r/s",
            col2,
            62,
            yellow,
            font_med,
        )
        put(f"s_r     : {telemetry.get('s_r', 0):+.4f}", col1, 84, red, font_med)
        put(f"s_l     : {telemetry.get('s_l', 0):+.4f}", col2, 84, red, font_med)

        # Gap summary row
        n_gaps = len(gaps)
        n_pass = sum(1 for g in gaps if g["passable"])
        bias = telemetry.get("steering_bias", 0.0)
        clear = telemetry.get("road_clear", False)
        if bias > 0:
            bias_str = f"bias=L({bias:.1f})"
        elif bias < 0:
            bias_str = f"bias=R({bias:.1f})"
        else:
            bias_str = "bias=0"
        state_str = "CLEAR" if clear else "OBS"
        gap_str = f"Gaps:{n_gaps} pass:{n_pass} {bias_str} [{state_str}]"
        gap_color = green if clear else (cyan if n_pass > 0 else red)
        put(gap_str, col1, 106, gap_color, font_small)

        n_pts = len(self._prev_pts) if self._prev_pts is not None else 0
        put(
            f"LK pts  : {n_pts}   Flow pairs: {len(flow_pairs)}",
            col1,
            120,
            white,
            font_small,
        )
        put("Press Q to quit", col2, 120, (120, 120, 120), font_small)

        cv2.line(display, (0, panel_h), (W * self.scale, panel_h), (60, 60, 60), 1)
        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        self._prev_gray = gray.copy()
        self._step += 1
        return key != ord("q")

    def close(self):
        cv2.destroyWindow(self.window_name)


# =============================================================================
# OTSU THRESHOLDING  (from scratch)
# =============================================================================


def otsu_threshold(gray: np.ndarray):
    """
    Otsu's method: maximise between-class variance σ²_B(t).
    Returns (thresh, binary) where binary is 255 for foreground pixels.
    """
    hist = np.zeros(256, dtype=np.float64)
    for val in gray.ravel():
        hist[int(val)] += 1.0
    hist /= hist.sum()

    cum_w = np.cumsum(hist)
    cum_mu = np.cumsum(hist * np.arange(256, dtype=np.float64))
    mu_total = cum_mu[-1]

    best_t, best_var = 0, -1.0
    for t in range(1, 255):
        w0 = cum_w[t]
        w1 = 1.0 - w0
        if w0 < 1e-9 or w1 < 1e-9:
            continue
        mu0 = cum_mu[t] / w0
        mu1 = (mu_total - cum_mu[t]) / w1
        var_b = w0 * w1 * (mu0 - mu1) ** 2
        if var_b > best_var:
            best_var = var_b
            best_t = t

    binary = (gray >= best_t).astype(np.uint8) * 255
    return best_t, binary


# =============================================================================
# GAUSSIAN BLUR  (separable, from scratch)
# =============================================================================


def _gaussian_1d(sigma: float) -> np.ndarray:
    size = max(3, int(6 * sigma + 1) | 1)
    x = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_1d(sigma)
    out = image.astype(np.float64)
    out = np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), 1, out)
    out = np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, out)
    return out


# =============================================================================
# SPATIAL GRADIENT  (central differences, from scratch)
# =============================================================================


def image_gradient(image: np.ndarray):
    gx = np.zeros_like(image, dtype=np.float64)
    gy = np.zeros_like(image, dtype=np.float64)
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    return gx, gy


# =============================================================================
# GAP ANALYSER
# =============================================================================


class GapAnalyser:
    """
    Determines whether each lateral gap (between obstacle edges and the road
    walls) is wide enough for the car body to pass through.

    Algorithm
    ---------
    1. Column projection
       Collapse the Otsu binary mask (road region only) along rows:
           col_occupied[j] = 1  if any pixel in column j is obstacle (255)
                           = 0  otherwise
       This gives a 1D occupancy profile of the road width.

    2. Gap enumeration (run-length encoding)
       Walk the 1D profile left-to-right. Each contiguous run of zeros
       (clear columns) bounded by either a wall or an obstacle is a candidate
       gap.  Runs shorter than GAP_MIN_COL_FRAC × image_width are discarded
       as noise.

       Up to three gap types exist:
         • left gap   : between left wall (col 0) and the left obstacle edge
         • inter gap  : between two separate obstacle blobs (multi-cone scene)
         • right gap  : between the right obstacle edge and right wall (col W-1)

    3. Pixel → world width conversion (pinhole model)
       The image was captured with a known horizontal FOV.  For a feature at
       assumed depth d = GAP_PROJ_DIST metres ahead:

           f_x   = (W / 2) / tan(HFOV/2)          pixel focal length
           y_L   = (col_left  - W/2) / f_x * d     world y of gap left edge
           y_R   = (col_right - W/2) / f_x * d     world y of gap right edge
           gap_world_width = |y_R - y_L|

       (The sign convention matches the car frame: positive y = left side.)

    4. Passability test
           passable = gap_world_width >= CAR_BODY_WIDTH + 2 * GAP_MARGIN

    5. Force modulation weights
       For each horizontal half of the image (left / right of centre):
         • If ALL gaps in that half are impassable  → weight = GAP_BLOCK_SCALE
         • Otherwise                                → weight = 1.0
       These weights multiply the lateral obstacle force computed from the
       centroid offset in APFPlanner.obstacle_repulsion.

    Parameters
    ----------
    img_w          : image width in pixels (must match the camera)
    car_body_width : lateral car extent (m)
    margin         : per-side safety clearance (m)
    hfov_deg       : horizontal FOV of the camera (degrees)
    proj_dist      : assumed obstacle distance for width projection (m)
    block_scale    : amplification factor for blocked corridor forces
    min_col_frac   : minimum gap width as fraction of image width (noise filter)
    """

    def __init__(
        self,
        img_w=CAM_IMG_W,
        car_body_width=CAR_BODY_WIDTH,
        margin=GAP_MARGIN,
        hfov_deg=GAP_HFOV_DEG,
        proj_dist=GAP_PROJ_DIST,
        block_scale=GAP_BLOCK_SCALE,
        min_col_frac=GAP_MIN_COL_FRAC,
    ):
        self.img_w = img_w
        self.car_body_width = car_body_width
        self.margin = margin
        self.proj_dist = proj_dist
        self.block_scale = block_scale
        self.min_col_frac = min_col_frac
        self.min_gap_cols = max(1, int(min_col_frac * img_w))

        # Pixel focal length from horizontal FOV
        hfov_rad = math.radians(hfov_deg)
        self._fx = (img_w / 2.0) / math.tan(hfov_rad / 2.0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _col_to_world_y(self, col: float) -> float:
        """
        Convert a pixel column index to a lateral world coordinate (metres).
        Column 0 → left edge of the camera frustum.
        Column W/2 → directly ahead (y = 0).

        Pinhole model at depth GAP_PROJ_DIST:
            y_world = (col - W/2) / f_x * d

        Note: image-x and world-y are anti-parallel in the standard car frame
        (image left = world +y, image right = world −y).  Negating col_offset
        restores this convention.
        """
        col_offset = col - self.img_w / 2.0
        # Negate so image-right → world -y
        return -col_offset / self._fx * self.proj_dist

    def _enumerate_gaps(self, col_occupied: np.ndarray):
        """
        Run-length encode the 1D column occupancy vector into a list of gaps.

        Each gap is a dict:
            {
              "col_left"  : int,   first clear column
              "col_right" : int,   last clear column (inclusive)
              "type"      : str,   "left_gap" | "right_gap" | "inter_gap"
            }

        Walls are treated as infinite obstacles at col = -1 and col = W.
        A gap that starts at col 0 is a left gap; one that ends at col W-1
        is a right gap; one flanked by obstacles on both sides is an inter gap.
        """
        W = len(col_occupied)
        gaps = []
        i = 0

        while i < W:
            if col_occupied[i] == 0:
                # Start of a clear run
                run_start = i
                while i < W and col_occupied[i] == 0:
                    i += 1
                run_end = i - 1  # inclusive

                run_len = run_end - run_start + 1
                if run_len < self.min_gap_cols:
                    continue  # too short — skip noise

                # Classify by position
                left_is_wall = run_start == 0
                right_is_wall = run_end == W - 1

                if left_is_wall and right_is_wall:
                    gap_type = "full_clear"  # no obstacle at all
                elif left_is_wall:
                    gap_type = "left_gap"
                elif right_is_wall:
                    gap_type = "right_gap"
                else:
                    gap_type = "inter_gap"

                gaps.append(
                    {
                        "col_left": run_start,
                        "col_right": run_end,
                        "type": gap_type,
                    }
                )
            else:
                i += 1

        return gaps

    # ── Public interface ──────────────────────────────────────────────────────

    def analyse(self, road_binary: np.ndarray):
        """
        Analyse gaps in the binary obstacle mask (road region only).

        Parameters
        ----------
        road_binary : (H_road, W) uint8 — Otsu mask cropped to road rows.
                      255 = obstacle, 0 = clear.

        Returns
        -------
        gaps : list of dicts, each containing:
               "col_left"    : int   — leftmost clear column of the gap
               "col_right"   : int   — rightmost clear column of the gap
               "type"        : str   — "left_gap" / "right_gap" / "inter_gap" / "full_clear"
               "world_left"  : float — world y of gap left edge (m)
               "world_right" : float — world y of gap right edge (m)
               "world_width" : float — gap lateral width in world space (m)
               "passable"    : bool  — True if car can fit through

        steering_bias : float
               +block_scale → steer left  (only left gap is passable)
               −block_scale → steer right (only right gap is passable)
               0.0          → no gap-based preference (centroid decides)
        """
        W = road_binary.shape[1]

        # Step 1: column projection — any obstacle pixel in the column?
        col_occupied = (road_binary.max(axis=0) > 0).astype(np.uint8)

        # Step 2: enumerate gaps
        gaps = self._enumerate_gaps(col_occupied)

        # Step 3: project each gap to world width and test passability
        min_passable_width = self.car_body_width + 2.0 * self.margin

        for g in gaps:
            # image col → world y (note anti-parallel axis, handled in helper)
            wy_l = self._col_to_world_y(g["col_left"])
            wy_r = self._col_to_world_y(g["col_right"])

            # world_width is always positive (take abs regardless of sign conv)
            g["world_left"] = min(wy_l, wy_r)
            g["world_right"] = max(wy_l, wy_r)
            g["world_width"] = abs(wy_r - wy_l)
            g["passable"] = g["world_width"] >= min_passable_width

        # Step 4: derive per-gap steering bias
        #
        # We want to answer: "which DIRECTION should the car steer to avoid
        # the obstacle?"  The answer comes directly from gap passability:
        #
        #   • left_gap  passable, right_gap impassable  → steer LEFT  (+y)
        #   • right_gap passable, left_gap  impassable  → steer RIGHT (−y)
        #   • both passable                             → no bias (centroid decides)
        #   • neither  passable                         → no bias (best effort)
        #
        # We return a signed steering_bias ∈ [−block_scale, +block_scale]:
        #   positive → car should go left  (+y world) → add to F_total_y
        #   negative → car should go right (−y world) → subtract from F_total_y
        #
        # This replaces the per-half weight multiplication which was fragile
        # when the car was already offset (2nd obstacle case).

        left_gaps = [g for g in gaps if g["type"] in ("left_gap", "full_clear")]
        right_gaps = [g for g in gaps if g["type"] in ("right_gap", "full_clear")]

        left_passable = any(g["passable"] for g in left_gaps) if left_gaps else False
        right_passable = any(g["passable"] for g in right_gaps) if right_gaps else False

        if left_passable and not right_passable:
            steering_bias = +self.block_scale  # steer left
        elif right_passable and not left_passable:
            steering_bias = -self.block_scale  # steer right
        else:
            steering_bias = 0.0  # centroid force decides

        return gaps, steering_bias


# =============================================================================
# APF PLANNER
# =============================================================================


class APFPlanner:
    """
    Three-component Artificial Potential Field with gap-clearance modulation.

    Components
    ----------
    F_att  : goal attraction (unit vector × α)
    F_obs  : obstacle repulsion (Otsu centroid offset × γ × gap_weights)
    F_road : road boundary repulsion (Morse potential)

    Gap clearance
    -------------
    GapAnalyser runs on every frame.  If only one side has a passable gap,
    the lateral repulsion toward the blocked side is scaled by GAP_BLOCK_SCALE,
    making atan2(F_total_y, F_total_x) naturally steer toward the open lane.
    """

    def __init__(
        self,
        goal_x=GOAL_X,
        goal_y=GOAL_Y,
        alpha=APF_ALPHA,
        A=APF_A,
        b=APF_B,
        gamma=APF_GAMMA,
        road_safe=ROAD_SAFE,
    ):
        self.goal = np.array([goal_x, goal_y], dtype=np.float64)
        self.alpha = alpha
        self.A = A
        self.b = b
        self.gamma = gamma
        self.y_r = road_safe
        self.y_l = -road_safe
        self._gap = GapAnalyser()

    # ── 1. Goal attraction ────────────────────────────────────────────────────

    def goal_attraction(self, car_x: float, car_y: float):
        """Unit vector toward goal, scaled by α."""
        dx = self.goal[0] - car_x
        dy = self.goal[1] - car_y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0, 0.0
        return self.alpha * dx / dist, self.alpha * dy / dist

    # ── 2. Obstacle repulsion from image ─────────────────────────────────────

    def obstacle_repulsion(self, gray: np.ndarray, car_x: float = 0.0):
        """
        Obstacle repulsion via Otsu blob centroid, with gap-clearance steering bias.

        Centroid force (unchanged from base)
        ─────────────────────────────────────
            prox_weight = obstacle_pixel_fraction          (proximity proxy)
            obs_fy_raw  = γ · prox_weight · cx_offset      (lateral, image frame)
            obs_fx      = −γ · prox_weight · cy_offset     (forward)

        Image→world axis: image-right (cx_offset > 0) = world −y (right of car).
        Subtracting obs_fy_raw from total_y pushes the car away from the centroid.

        Gap clearance steering bias
        ────────────────────────────
        GapAnalyser classifies left/right gaps as passable or not and returns a
        single signed steering_bias:

            steering_bias > 0  → only left  gap passable → add  to total_y → steer left
            steering_bias < 0  → only right gap passable → sub from total_y → steer right
            steering_bias = 0  → both or neither passable → centroid decides

        The bias is injected directly into total_y in APFPlanner.compute() rather
        than scaling obs_fy.  This avoids the sign-inversion and cross-axis
        confusion that plagued the per-half weight multiplier approach, and keeps
        the centroid force intact so forward repulsion (obs_fx) still works.

        Why this fixes the 2nd-obstacle failure
        ─────────────────────────────────────────
        When the car is already offset from centre (after swerving for cone 1),
        the overall centroid of the new cone may appear near the image centre,
        giving a near-zero cx_offset and almost no lateral force.  The steering
        bias bypasses the centroid entirely — it reads which corridor is open from
        the 1D column profile, which works correctly regardless of car lateral
        position.

        Returns
        -------
        obs_fx         : float     world-frame forward  repulsion
        obs_fy         : float     world-frame lateral  repulsion (centroid only)
        binary         : ndarray   Otsu mask (H,W) for debug
        gaps           : list      gap dicts from GapAnalyser
        steering_bias  : float     signed gap-clearance bias (+= steer left)
        """
        H, W = gray.shape
        crop_start = int(H * OBS_CROP_FRAC)

        # ── Full-frame Otsu + polarity check ──────────────────────────────
        #
        # Running Otsu on only the road crop fails: the road surface fills
        # most of the crop with a near-uniform intensity, giving a unimodal
        # histogram whose best "split" is arbitrary — the entire crop ends
        # up as one class, so either all columns are occupied (gaps=0) or
        # all are clear.
        #
        # Running Otsu on the FULL frame works because the sky (top 1/3) is
        # distinctly brighter than the road, creating a genuine bimodal
        # histogram. The threshold reliably separates bright (sky + yellow
        # cone top) from dark (road + black cone stripes).
        #
        # After thresholding we:
        #   1. Crop to the road region (discard sky rows).
        #   2. Check polarity: if the bottom floor rows are mostly "foreground"
        #      the mask is inverted (road itself was called obstacle).
        #   3. Erode with a small kernel to kill 1-2 px road-marking noise.

        _, binary_full = otsu_threshold(gray)  # full-frame Otsu
        road_binary = binary_full[crop_start:, :]  # crop to road rows

        # Polarity check — bottom floor rows must be mostly clear (road)
        H_road = road_binary.shape[0]
        floor_rows = max(1, int(H_road * OBS_FLOOR_FRAC))
        floor_mean = road_binary[-floor_rows:, :].mean() / 255.0
        if floor_mean > 0.5:  # road is being called foreground
            road_binary = 255 - road_binary  # invert

        # Morphological erosion — remove isolated noise pixels
        if OBS_ERODE_KSIZE > 1:
            k = np.ones((OBS_ERODE_KSIZE, OBS_ERODE_KSIZE), np.uint8)
            road_binary = cv2.erode(road_binary, k, iterations=1)

        # Full-frame binary for debug display
        binary = np.zeros_like(gray)
        binary[crop_start:, :] = road_binary

        # End-wall gate: in the final GOAL_CLEAR_DIST metres the end wall fills
        # the camera and is detected as an obstacle. Treat this zone as clear so
        # the lane spring can fire and the car straightens up for the finish.
        if car_x >= GOAL_X - GOAL_CLEAR_DIST:
            return 0.0, 0.0, binary, [], True, False

        # ── Gap analysis (image-based, passed back raw) ───────────────────
        # steering_bias is NOT applied here — it is computed in compute()
        # based on world distance to the next cone, independently of pixel
        # proximity.  This prevents the proximity gate from silencing the
        # gap bias when the cone is still far but the corridor is blocked.
        gaps, gap_bias_image = self._gap.analyse(road_binary)

        # ── Centroid-based lateral repulsion ──────────────────────────────
        normalised = road_binary.astype(np.float64) / 255.0
        sigma = max(1.0, W / OBS_SIGMA_DIV)
        smoothed = gaussian_blur(normalised, sigma=sigma)
        total_mass = smoothed.sum()

        if total_mass < 1e-6:
            return 0.0, 0.0, binary, gaps, True, False

        H_road, W_road = smoothed.shape
        half_w = W_road / 2.0

        cols = np.arange(W_road, dtype=np.float64)
        rows = np.arange(H_road, dtype=np.float64)
        cx = float((smoothed * cols[np.newaxis, :]).sum() / total_mass)

        cx_offset = (cx - half_w) / half_w
        prox_weight = float((road_binary > 0).sum()) / (H_road * W_road)

        # Proximity gate: centroid force only fires when cone is close enough.
        # Gap bias is handled separately in compute() — not affected by this gate.
        if prox_weight < OBS_MIN_PROX:
            return 0.0, 0.0, binary, gaps, True, False

        obs_fy = self.gamma * prox_weight * cx_offset
        road_clear = False
        need_slow = prox_weight >= OBS_SLOW_PROX

        return 0.0, obs_fy, binary, gaps, road_clear, need_slow

    # ── 3. Road boundary repulsion (Morse potential) ──────────────────────────

    def road_repulsion(self, car_y: float) -> float:
        """
        Two-zone wall repulsion:

        Zone 1 — buffer  ROAD_SAFE < |y| ≤ WALL_CONTACT_Y  (0.50 – 0.80 m)
            Quadratic: F = APF_A · (penetration / buffer)²
            Grows from 0 to APF_A over the 0.30 m buffer.

        Zone 2 — danger  |y| > WALL_CONTACT_Y  (car edge at / past wall)
            Exponential: F = APF_A · exp(k · excess)   k = 10
            Grows rapidly from APF_A upward so the car literally cannot stay
            outside WALL_CONTACT_Y — force doubles every 0.07 m beyond it.

        Force profile (APF_A=9):
            |y|=0.50  →   0.0   (dead-zone edge)
            |y|=0.65  →   2.2   (wall-override zone)
            |y|=0.80  →   9.0   (car touches wall — zone 2 starts)
            |y|=0.90  →  24.3   (0.10 m past wall)
            |y|=1.00  →  66.7   (0.20 m past wall — overwhelms everything)
        """
        penetration = abs(car_y) - ROAD_SAFE
        if penetration <= 0.0:
            return 0.0
        buffer = WALL_CONTACT_Y - ROAD_SAFE  # 0.30 m
        if penetration <= buffer:
            # Zone 1: quadratic
            ratio = penetration / buffer
            magnitude = self.A * ratio**2
        else:
            # Zone 2: exponential — grows steeply past wall-contact point
            excess = penetration - buffer  # how far past WALL_CONTACT_Y
            magnitude = self.A * math.exp(10.0 * excess)
        return math.copysign(magnitude, car_y)

    # ── Total force → desired heading ─────────────────────────────────────────

    def compute(self, car_x: float, car_y: float, gray: np.ndarray):
        """
        Sum all APF components → desired heading ψ_d.

        Force budget
        ─────────────
            total_x = F_att_x − F_obs_x
            total_y = F_att_y − F_obs_y − F_road_y + steering_bias + F_lane_y

        steering_bias : signed gap-clearance bias (positive = steer left)
        F_lane_y      : lane-centre spring −LANE_K·car_y, active only when
                        road is obstacle-free.  Pulls the car back to y=0
                        after each swerve so it is correctly centred when
                        it approaches the next obstacle.

        Returns
        -------
        psi_d          : float     desired heading (radians)
        binary         : ndarray   Otsu mask for debug
        forces         : tuple     (att_x, att_y, obs_x, obs_y, road_y)
        gaps           : list      gap dicts from GapAnalyser
        steering_bias  : float     gap-clearance lateral bias
        road_clear     : bool      True when no obstacle detected this frame
        """
        att_x, att_y = self.goal_attraction(car_x, car_y)
        obs_x, obs_y, binary, gaps, road_clear, need_slow = self.obstacle_repulsion(
            gray, car_x
        )
        road_y = self.road_repulsion(car_y)

        upcoming_cones = sorted([cx for cx in CONE_X_POSITIONS if cx > car_x])
        past_cones = sorted([cx for cx in CONE_X_POSITIONS if cx <= car_x])

        # ── Steering bias (world-distance + cone_y fallback) ─────────────
        steering_bias = 0.0
        if upcoming_cones and (upcoming_cones[0] - car_x) <= GAP_TRIGGER_DIST:
            next_idx = CONE_X_POSITIONS.index(upcoming_cones[0])
            next_cy = CONE_Y_POSITIONS[next_idx]

            # Lateral clearance check: if the car has already moved far enough
            # away from the cone in the avoidance direction, cancel the bias.
            # Prevents the bias from pushing the car all the way to the wall.
            lateral_sep = abs(
                car_y - next_cy
            )  # distance between car centre and cone centre
            already_clear = lateral_sep >= BIAS_CLEAR_MARGIN

            if not already_clear:
                left_gaps = [g for g in gaps if g["type"] in ("left_gap", "full_clear")]
                right_gaps = [
                    g for g in gaps if g["type"] in ("right_gap", "full_clear")
                ]
                left_pass = (
                    any(g["passable"] for g in left_gaps) if left_gaps else False
                )
                right_pass = (
                    any(g["passable"] for g in right_gaps) if right_gaps else False
                )
                if left_pass and not right_pass:
                    steering_bias = +GAP_BLOCK_SCALE
                elif right_pass and not left_pass:
                    steering_bias = -GAP_BLOCK_SCALE
                else:
                    steering_bias = -math.copysign(GAP_BLOCK_SCALE, next_cy)

        # ══════════════════════════════════════════════════════════════════
        # THREE-MODE FORCE ARCHITECTURE — strictly no overlap
        # ══════════════════════════════════════════════════════════════════
        # MODE 1 — WALL OVERRIDE  |car_y| > WALL_OVERRIDE_Y (0.65 m)
        #   Only road repulsion drives total_y. Everything else is zero.
        #   Heading clamp expands to ±60° so the car can actually turn back.
        #   Exits automatically once |car_y| ≤ WALL_OVERRIDE_Y.
        #
        # MODE 2 — OBSTACLE ACTIVE  (obstacle in camera OR in trigger zone)
        #   Attraction forward + lateral repulsion + steering_bias.
        #   Road wall repulsion excluded (car is inside safe zone here).
        #   Lane spring strictly OFF — don't fight avoidance.
        #
        # MODE 3 — FREE  (no obstacle, past last cone, not yet approaching next)
        #   Zero lateral forces. Lane spring pulls y→0 ONLY if |y|>LANE_DEADZONE.
        #   Strictly zero force when y≈0 — eliminates all wiggle.
        # ══════════════════════════════════════════════════════════════════

        in_wall_zone = abs(car_y) > WALL_OVERRIDE_Y
        obstacle_active = (not road_clear) or (steering_bias != 0.0)

        if in_wall_zone:
            # MODE 1
            total_x = att_x
            total_y = -road_y  # road_y sign = car_y sign → -road_y toward centre
            lane_y = 0.0

        elif obstacle_active:
            # MODE 2
            total_x = att_x - obs_x
            total_y = att_y - obs_y + steering_bias
            lane_y = 0.0

        else:
            # MODE 3
            total_x = att_x
            past_cleared = (not past_cones) or (
                car_x > past_cones[-1] + CAR_HALF_LENGTH
            )
            next_far_enough = (not upcoming_cones) or (
                upcoming_cones[0] - car_x > GAP_TRIGGER_DIST
            )
            if past_cleared and next_far_enough and abs(car_y) > LANE_DEADZONE:
                clamped_y = max(-LANE_CLAMP, min(LANE_CLAMP, car_y))
                lane_y = -LANE_K * clamped_y
            else:
                lane_y = 0.0
            total_y = lane_y

        psi_d_raw = math.atan2(total_y, total_x)

        # Heading clamp: ±30° normally, expands to ±60° near/at wall
        wall_prox = max(0.0, abs(car_y) - ROAD_SAFE) / (WALL_CONTACT_Y - ROAD_SAFE)
        psi_limit = MAX_PSI_D + (MAX_PSI_D_WALL - MAX_PSI_D) * min(wall_prox, 1.0)
        psi_d = max(-psi_limit, min(psi_limit, psi_d_raw))

        return (
            psi_d,
            binary,
            (att_x, att_y, obs_x, obs_y, road_y),
            gaps,
            steering_bias,
            road_clear,
            lane_y,
            need_slow,
        )


# =============================================================================
# GRADIENT TRACKING SLIDING MODE CONTROLLER  (lateral)
# =============================================================================


class SlidingModeController:
    """
    Lateral steering controller — paper Section VI-B.

        s_r = c_r · ψ_e + dψ_e/dt
        u   = −u0 · sign(s_r)
        δ_f += u · dt
    """

    def __init__(self, c_r=SMC_CR, u0=SMC_U0):
        self.c_r = c_r
        self.u0 = u0
        self._delta = 0.0
        self._prev_psi_e = 0.0

    def step(self, psi: float, psi_d: float, dt: float) -> float:
        psi_e = (psi - psi_d + math.pi) % (2.0 * math.pi) - math.pi
        dpsi_e = (psi_e - self._prev_psi_e) / dt if dt > 0 else 0.0
        s_r = self.c_r * psi_e + dpsi_e
        u = -self.u0 * math.copysign(1.0, s_r)
        self._delta += u * dt
        self._delta = max(-MAX_STEER, min(MAX_STEER, self._delta))
        self._prev_psi_e = psi_e
        return self._delta


# =============================================================================
# LONGITUDINAL SLIDING MODE CONTROLLER
# =============================================================================


class SpeedController:
    """
    Longitudinal speed controller — paper Section VI-B.

        s_l = c_l · (v − v_d)
        a   = −a0 · sign(s_l)
        wheel_speed += a · dt
    """

    def __init__(
        self, c_l=SPD_CL, a0=SPD_A0, v_d=TARGET_SPEED, max_wheel_speed=MAX_WHEEL_SPEED
    ):
        self.c_l = c_l
        self.a0 = a0
        self.v_d = v_d
        self.max_wheel_speed = max_wheel_speed
        self._wheel_speed = 0.0

    def step(self, v: float, dt: float, v_d_override: float = None) -> float:
        target = v_d_override if v_d_override is not None else self.v_d
        s_l = self.c_l * (v - target)
        a = -self.a0 * math.copysign(1.0, s_l)
        self._wheel_speed += a * dt
        self._wheel_speed = max(0.0, min(self.max_wheel_speed, self._wheel_speed))
        return self._wheel_speed


# =============================================================================
# ACTUATION HELPERS
# =============================================================================


def set_steering(car_id, joints, angle: float):
    angle = max(-MAX_STEER, min(MAX_STEER, angle))
    for j in joints:
        p.setJointMotorControl2(
            car_id, j, p.POSITION_CONTROL, targetPosition=angle, force=10
        )


def set_throttle(car_id, joints, speed: float):
    for j in joints:
        p.setJointMotorControl2(
            car_id, j, p.VELOCITY_CONTROL, targetVelocity=speed, force=WHEEL_FORCE
        )


# =============================================================================
# MAIN LOOP
# =============================================================================


def run_controller():
    car_id, steering_joints, motor_joints = setup_simulation(gui=True)

    camera = CameraHelper(car_id)
    planner = APFPlanner()
    smc = SlidingModeController()
    speed_c = SpeedController()
    viz = DebugVisualizer()
    dt = SIM_DT

    print("[Controller] APF + Sliding Mode + Gap Clearance controller started.")
    print(
        f"  Road physical : ±{ROAD_HALF} m  car_half_w={CAR_HALF_WIDTH} m  wall_contact=±{WALL_CONTACT_Y} m"
    )
    print(
        f"  Road safe APF : ±{ROAD_SAFE} m  (repulsion dead-zone, buffer={WALL_CONTACT_Y-ROAD_SAFE:.2f} m to wall contact)"
    )
    print(f"  Goal          : ({GOAL_X}, {GOAL_Y})")
    print(f"  APF gains     : α={APF_ALPHA}  A={APF_A}  b={APF_B}  γ={APF_GAMMA}")
    print(f"  Car body width: {CAR_BODY_WIDTH} m  margin={GAP_MARGIN} m/side")
    print(
        f"  Min passable  : {CAR_BODY_WIDTH + 2*GAP_MARGIN:.2f} m  block_scale={GAP_BLOCK_SCALE}"
    )
    print(f"  Gap HFOV      : {GAP_HFOV_DEG}°  proj_dist={GAP_PROJ_DIST} m")
    print(
        f"  Lateral SMC   : c_r={SMC_CR}  u0={SMC_U0}  max_steer={MAX_STEER} rad  max_psi_d=±{math.degrees(MAX_PSI_D):.0f}°..±{math.degrees(MAX_PSI_D_WALL):.0f}° near wall"
    )
    print(f"  Speed SMC     : c_l={SPD_CL}  a0={SPD_A0}  v_d={TARGET_SPEED}")
    print(f"  Camera        : {CAM_IMG_W}×{CAM_IMG_H}  fov={CAM_FOV}°")
    print(
        f"  Detection     : full-frame Otsu + polarity check  floor={OBS_FLOOR_FRAC}  erode={OBS_ERODE_KSIZE}"
    )
    print(
        f"  Obs detection : min_prox={OBS_MIN_PROX}  slow_prox={OBS_SLOW_PROX}  slow_spd={OBS_SLOW_SPEED}  goal_clear={GOAL_CLEAR_DIST} m"
    )
    print(
        f"  Lane spring   : K={LANE_K}  clamp=±{LANE_CLAMP} m  deadzone=±{LANE_DEADZONE} m  car_half={CAR_HALF_LENGTH} m"
    )
    print(f"  LK            : levels={LK_LEVELS}  window={LK_WINDOW}  ε={LK_EPSILON}\n")

    prev_v = 0.0
    step = 0
    running = True

    try:
        while running:
            # ── 1. Sense ──────────────────────────────────────────────────
            pos, orn = p.getBasePositionAndOrientation(car_id)
            x, y, _ = pos
            _, _, yaw = p.getEulerFromQuaternion(orn)
            lin_vel, _ = p.getBaseVelocity(car_id)
            vx, vy, _ = lin_vel
            v = math.hypot(vx, vy)
            accel = (v - prev_v) / dt
            prev_v = v

            if x >= GOAL_X:
                print(f"[Controller] Goal reached at x={x:.2f} m — stopping.")
                set_throttle(car_id, motor_joints, 0.0)
                set_steering(car_id, steering_joints, 0.0)
                break

            # ── 2. Camera + APF + Gap analysis ───────────────────────────
            gray = camera.grab_gray()
            (
                psi_d,
                binary,
                forces,
                gaps,
                steering_bias,
                road_clear,
                lane_y_val,
                need_slow,
            ) = planner.compute(x, y, gray)

            # ── 3. Lateral SMC → steering ─────────────────────────────────
            delta = smc.step(yaw, psi_d, dt)

            # ── 4. Longitudinal SMC → throttle (slow near obstacles) ───────
            v_target = OBS_SLOW_SPEED if need_slow else None
            wheel_speed = speed_c.step(v, dt, v_d_override=v_target)

            # ── 5. Manifold values for display ────────────────────────────
            psi_e = (yaw - psi_d + math.pi) % (2 * math.pi) - math.pi
            s_r = smc.c_r * psi_e
            v_tgt = OBS_SLOW_SPEED if need_slow else TARGET_SPEED
            s_l = speed_c.c_l * (v - v_tgt)

            # ── 6. Debug visualizer ───────────────────────────────────────
            telemetry = {
                "speed": v,
                "yaw_deg": math.degrees(yaw),
                "steer_deg": math.degrees(delta),
                "psi_d_deg": math.degrees(psi_d),
                "accel": accel,
                "wheel_speed": wheel_speed,
                "s_r": s_r,
                "s_l": s_l,
                "gaps": gaps,
                "steering_bias": steering_bias,
                "road_clear": road_clear,
            }
            running = viz.update(gray, telemetry)

            # ── 7. Act ────────────────────────────────────────────────────
            set_steering(car_id, steering_joints, delta)
            set_throttle(car_id, motor_joints, wheel_speed)

            p.stepSimulation()
            time.sleep(dt)

            # ── Log every second ──────────────────────────────────────────
            if step % 60 == 0:
                att_x, att_y, obs_x, obs_y, road_y = forces
                n_pass = sum(1 for g in gaps if g["passable"])
                clamped_y = max(-LANE_CLAMP, min(LANE_CLAMP, y))
                gap_str = (
                    f"gaps={len(gaps)} pass={n_pass} "
                    f"bias={steering_bias:+.1f} "
                    f"lane={lane_y_val:+.3f} "
                    f"{'CLEAR' if road_clear else 'OBS'}"
                )
                print(
                    f"  x={x:6.2f}  y={y:+.3f}  ψ={math.degrees(yaw):+5.1f}° | "
                    f"ψ_d={math.degrees(psi_d):+.1f}°  "
                    f"ψ_e={math.degrees(psi_e):+.1f}°  "
                    f"s_r={s_r:+.3f}  δ={math.degrees(delta):+.1f}° | "
                    f"v={v:.2f}m/s  a={accel:+.3f}  s_l={s_l:+.3f}  w={wheel_speed:.1f} | "
                    f"F_att=({att_x:+.2f},{att_y:+.3f})  "
                    f"F_obs=({obs_x:+.4f},{obs_y:+.4f})  "
                    f"F_road={road_y:+.3f} | "
                    f"{gap_str}"
                )
            step += 1

    except KeyboardInterrupt:
        print("\n[Controller] Interrupted.")
    finally:
        viz.close()
        try:
            p.disconnect()
        except Exception:
            pass
        print("[Controller] Disconnected.")


if __name__ == "__main__":
    run_controller()

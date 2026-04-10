import cv2
import numpy as np

# ---------------------LucasKanade basic functions---------------------------------------------------


class LKSupport:

    def build_pyramid(img, levels=4):
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

            if (
                xi - half_w < 0
                or xi + half_w >= w
                or yi - half_w < 0
                or yi + half_w >= h
            ):
                status[i] = 0
                continue

            Ix_win = Ix[
                yi - half_w : yi + half_w + 1, xi - half_w : xi + half_w + 1
            ].ravel()
            Iy_win = Iy[
                yi - half_w : yi + half_w + 1, xi - half_w : xi + half_w + 1
            ].ravel()

            A = np.vstack([Ix_win, Iy_win]).T
            ATA = A.T @ A

            eigenvalues = np.linalg.eigvalsh(ATA)
            if eigenvalues[0] < 1e-3:
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

                patch1 = im1f[
                    yi - half_w : yi + half_w + 1, xi - half_w : xi + half_w + 1
                ]
                patch2 = im2f[
                    nyi - half_w : nyi + half_w + 1, nxi - half_w : nxi + half_w + 1
                ]

                It_win = (patch2 - patch1).flatten()
                b = -A.T @ It_win
                delta = ATA_inv @ b
                vx += delta[0]
                vy += delta[1]

                if np.linalg.norm(delta) < epsilon:
                    break

            if status[i]:
                p1[i, 0, 0] = x0 + vx
                p1[i, 0, 1] = y0 + vy

        return p1, status


# ---------------------Optical Flow using Lucas Kanade pyramidal algo--------------------------------------------


class Opticalflow:

    def pyramidal_lucas_kanade(
        im1, im2, p0, levels=3, window_size=15, max_iter=20, epsilon=0.01
    ):
        pyr1 = LKSupport.build_pyramid(im1, levels)
        pyr2 = LKSupport.build_pyramid(im2, levels)

        flow = np.zeros_like(p0, dtype=np.float32)
        status = np.ones(len(p0), dtype=np.uint8)

        for lvl in range(levels - 1, -1, -1):
            scale = 2**lvl
            p_lvl = p0.astype(np.float32) / scale
            p_guess = p_lvl + flow

            p1_lvl, st = LKSupport.lk_single_level(
                pyr1[lvl],
                pyr2[lvl],
                p_guess,
                window_size=window_size,
                max_iter=max_iter,
                epsilon=epsilon,
            )

            delta = p1_lvl - p_guess
            flow = flow + delta
            status &= st

            if lvl > 0:
                flow *= 2

        p1 = p0.astype(np.float32) + flow
        return p1, status

    def draw_trails(trail_overlay, p0_arr, p1_arr):
        for old_pt, new_pt in zip(p0_arr, p1_arr):
            x0, y0 = int(old_pt[0]), int(old_pt[1])
            x1, y1 = int(new_pt[0]), int(new_pt[1])
            dist = np.hypot(x1 - x0, y1 - y0)

            if dist < 0.5:
                continue

            angle = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 360
            speed = min(dist / 10.0, 1.0)

            hsv_color = np.uint8([[[int(angle / 2), 120, int(200 * speed)]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()

            cv2.line(trail_overlay, (x0, y0), (x1, y1), bgr_color, 2)
            cv2.circle(trail_overlay, (x1, y1), 2, bgr_color, -1)


# ── Config ──────────────────────────────────────────


class config:

    SCALE = 0.33
    MAX_CORNERS = 100
    QUALITY_LEVEL = 0.03
    MIN_DISTANCE = 20
    LEVELS = 2
    WIN_SIZE = 15
    MAX_ITER = 30
    TRAIL_FADE = 0.8
    JUMP_THRESH = 40
    FRAME_SKIP = 2


# ── Setup ────────────────────────────────────────────


class Video_InputandOutput:

    def __init__(self, Videoname):

        self.Videoname = Videoname

    def Process_Video(self):

        cap = cv2.VideoCapture(self.Videoname)

        if not cap.isOpened():
            print("ERROR: Could not open video file.")
            print("Make sure OF_trimmed.mp4 is in the same folder as this script.")
            exit()

        print(
            f"Video opened. FPS: {cap.get(cv2.CAP_PROP_FPS)}, "
            f"Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}, "
            f"Size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
        )

        ret, old_frame = cap.read()
        if not ret:
            print("ERROR: Could not read first frame.")
            exit()

        old_frame = cv2.resize(old_frame, (0, 0), fx=config.SCALE, fy=config.SCALE)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        trail_overlay = np.zeros_like(old_frame, dtype=np.float32)
        p0 = cv2.goodFeaturesToTrack(
            old_gray, config.MAX_CORNERS, config.QUALITY_LEVEL, config.MIN_DISTANCE
        )
        frame_count = 0

        print("Starting optical flow. Press Q to quit.")

        # ADD HERE
        cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Optical Flow", 1280, 720)

        # ── Main loop ────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended.")
                break

            frame_count += 1
            if frame_count % config.FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (0, 0), fx=config.SCALE, fy=config.SCALE)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if p0 is not None and len(p0) > 0:

                p1, status = Opticalflow.pyramidal_lucas_kanade(
                    old_gray,
                    frame_gray,
                    p0,
                    levels=config.LEVELS,
                    window_size=config.WIN_SIZE,
                    max_iter=config.MAX_ITER,
                )

                good_mask = status == 1

                if good_mask.any():
                    motion = np.linalg.norm(
                        p1[good_mask].reshape(-1, 2) - p0[good_mask].reshape(-1, 2),
                        axis=1,
                    )
                    motion_mask = motion < config.JUMP_THRESH
                    idx = np.where(good_mask)[0][motion_mask]

                    p0_good = p0[idx].reshape(-1, 2)
                    p1_good = p1[idx].reshape(-1, 2)

                    trail_overlay *= config.TRAIL_FADE
                    Opticalflow.draw_trails(trail_overlay, p0_good, p1_good)

                    p0 = p1_good.reshape(-1, 1, 2)
                else:
                    p0 = None

            # Re-detect if too few points
            if p0 is None or len(p0) < 50:
                p0 = cv2.goodFeaturesToTrack(
                    frame_gray,
                    config.MAX_CORNERS,
                    config.QUALITY_LEVEL,
                    config.MIN_DISTANCE,
                )

            # Composite trails onto frame
            trail_uint8 = np.clip(trail_overlay, 0, 255).astype(np.uint8)
            output = cv2.add(frame, trail_uint8)

            cv2.imshow("Optical Flow", output)

            old_gray = frame_gray.copy()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit by user.")
                break

        # ── Cleanup ──────────────────────────────────────────
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


v = Video_InputandOutput("OF_trimmed.mp4")
v.Process_Video()

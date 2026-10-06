"""
Q1LiteVisualBridge: Q1 Lite eye cameras → fly ommatidium format.

Same pipeline as the Go2 bridge (unitree_bridge/go2_vision.py): renders
the forward-facing eye_left/eye_right cameras (defined on the base body
in q1lite.xml, looking along the body +Y forward axis with 90 deg FOV)
and grid-samples them into the (2, 721, 2) ommatidium arrays consumed by
visual_system.py (T2 → LC4/LPLC2 → Giant Fiber escape).

RealCameraVisualBridge: the same ommatidium contract, but the input is
the Raspberry Pi USB camera instead of MuJoCo offscreen renders — the
fly's eyes see the real world (fly_embodied_q1lite.py --camera real
--visual). The camera frame is split left/right into the two compound
eyes and sampled with the identical 721-point hexagonal grid.
"""

import numpy as np

from unitree_bridge.go2_vision import Go2VisualBridge


class Q1LiteVisualBridge(Go2VisualBridge):
    """Renders Q1 Lite eye_left/eye_right and converts to ommatidia.

    The Q1 Lite model defines eye_left/eye_right cameras facing forward
    (+Y) — the same contract as the Go2 eye cameras — so the Go2
    sampling pipeline is reused unchanged.
    """

    def __init__(self, model, data, width=128, height=128,
                 contrast_gain=0.85):
        print(f"[Q1LiteVisual] Q1 Lite eyes (forward +Y), "
              f"{width}x{height} render")
        super().__init__(model, data, width=width, height=height,
                         contrast_gain=contrast_gain)


class RealCameraVisualBridge(Go2VisualBridge):
    """Pi USB camera frame → compound-eye ommatidia (drop-in for
    Q1LiteVisualBridge when --camera real --visual).

    The latest camera frame is split vertically: the left half becomes
    the left eye, the right half the right eye (a coarse stereo split of
    the real scene). Each half is resized to width×height and sampled
    with the SAME 721-point hexagonal grid + 2-channel photoreceptor
    scheme as the rendered bridges, so visual_system.py (T2 → LC4 → GF)
    works unchanged.
    """

    def __init__(self, camera, width=128, height=128, contrast_gain=0.85):
        # Deliberately NOT calling super().__init__(): no MuJoCo model,
        # no offscreen renderer — the input is the live camera instead.
        self.camera = camera
        self.width = width
        self.height = height
        self.contrast_gain = contrast_gain
        self._ommatidia = 721
        self._grid = self._build_ommatidium_grid()
        self.last_rgb_left = None
        self.last_rgb_right = None
        print(f"[RealCameraVisual] Pi camera → compound eyes "
              f"(left/right frame split), {width}x{height}/eye, "
              f"{self._ommatidia} ommatidia/eye")

    def process(self):
        """Return ommatidium array from the latest real camera frame.

        Returns:
            vision_obs: np.ndarray (2, 721, 2) float32 in [0, 1] —
            [0] = left eye (left half of frame), [1] = right eye.
            Dark (zeros) while the camera is warming up.
        """
        # Lazy import: this module is imported at module level by
        # fly_embodied_q1lite.py, so a module-level cv2 import would drag
        # cv2's bundled SDL2 into every spawn'd child (brain monitor =
        # pygame's SDL2 → objc class-collision warnings).
        import cv2

        frame = self.camera.get_frame()      # BGR or None
        if frame is None:
            return np.zeros((2, self._ommatidia, 2), dtype=np.float32)

        h, w = frame.shape[:2]
        left_bgr = frame[:, :w // 2]
        right_bgr = frame[:, w // 2:]
        left_rgb = cv2.cvtColor(
            cv2.resize(left_bgr, (self.width, self.height),
                       interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(
            cv2.resize(right_bgr, (self.width, self.height),
                       interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB)

        gray_left = np.mean(left_rgb, axis=2).astype(np.float32) / 255.0
        gray_right = np.mean(right_rgb, axis=2).astype(np.float32) / 255.0

        omm_left = self._sample_ommatidia(gray_left)
        omm_right = self._sample_ommatidia(gray_right)

        self.last_rgb_left = left_rgb
        self.last_rgb_right = right_rgb

        vision_obs = np.stack([omm_left, omm_right], axis=0).astype(np.float32)

        # Same contrast-gain semantics as the rendered bridges
        if self.contrast_gain < 1.0:
            mean_val = vision_obs.mean()
            vision_obs = mean_val + self.contrast_gain * (vision_obs - mean_val)

        return vision_obs

    def get_eye_images(self):
        """Return last (rgb_left, rgb_right) uint8 images or (None, None)."""
        return (self.last_rgb_left, self.last_rgb_right)

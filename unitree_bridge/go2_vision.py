"""
Go2VisualBridge: Renders Go2 eye cameras and converts to ommatidium format
compatible with the fly-brain visual_system.py pipeline.

Replaces flygym's compound-eye retina (retina.correct_fisheye +
retina.raw_image_to_hex_pxls) with a simpler grid-sampling approach
that maps Go2 camera renders to (2, 721, 2) ommatidium arrays.

Pipeline:
  mujoco.Renderer (eye_left/eye_right) → grayscale → grid sample
  → (2, 721, 2) ommatidium array → VisualSystem.process_visual_layers()
  → T2 neuron rates → brain injection + escape threat detection
"""

import numpy as np
import mujoco


class Go2VisualBridge:
    """
    Renders Go2 forward-facing cameras and converts to ommatidium format.

    The Go2 has 2 cameras mounted on base_link (eye_left, eye_right),
    each with 90° FOV. Images are rendered at 256×256, converted to
    grayscale, and downsampled to 721 "ommatidia" per eye via grid sampling.

    Each ommatidium has 2 "photoreceptor channels" created by sampling
    both the mean and a slightly offset pixel to simulate the two
    photoreceptor subtypes (R1-R6 vs R7/R8) in the fly retina.
    """

    def __init__(self, model, data, width=256, height=256,
                 contrast_gain=0.85):
        """
        Args:
            model: mujoco.MjModel
            data:   mujoco.MjData
            width:  render width per eye
            height: render height per eye
            contrast_gain: gain applied to brightness before contrast calc.
                           1.0 = full contrast, <1.0 = suppress background,
                           reduces false T2 activation on static scenes.
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.contrast_gain = contrast_gain

        # Create offscreen renderer
        self._renderer = mujoco.Renderer(model, height=height, width=width)

        # Look up camera IDs
        self._cam_left_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, 'eye_left')
        self._cam_right_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, 'eye_right')

        if self._cam_left_id < 0 or self._cam_right_id < 0:
            raise RuntimeError("eye_left/eye_right cameras not found in Go2 model. "
                               "Make sure go2.xml has been updated with camera elements.")

        # Precompute ommatidium sampling grid
        self._ommatidia = 721
        self._grid = self._build_ommatidium_grid()

        print(f"[Go2Visual] Cameras: eye_left={self._cam_left_id}, "
              f"eye_right={self._cam_right_id}")
        print(f"[Go2Visual] Render: {width}×{height}, "
              f"{self._ommatidia} ommatidia/eye")

    def _build_ommatidium_grid(self):
        """
        Build sampling grid for 721 ommatidia per eye.

        Uses a hexagonal-like packing: concentric rings of points
        covering the image, reducing toward edges. Center-weighted
        to match the fly retina's higher acuity in the frontal field.
        """
        cx, cy = self.width / 2, self.height / 2
        radius = min(self.width, self.height) * 0.45

        # Concentric rings: more points in center, fewer at edges
        n_rings = 15
        points = []

        for ring in range(n_rings):
            r = radius * (ring + 1) / n_rings
            # Points per ring: roughly proportional to circumference
            n_pts = int(6 * (ring + 1))
            for j in range(n_pts):
                angle = 2 * np.pi * j / n_pts
                # Stagger alternate rings for hexagonal packing
                if ring % 2 == 1:
                    angle += np.pi / n_pts
                px = cx + r * np.cos(angle)
                py = cy + r * np.sin(angle)
                points.append((px, py))

        # Add center point
        points.append((cx, cy))

        # Trim to exactly 721
        points = points[:self._ommatidia]
        # If fewer than 721, pad with edge points
        while len(points) < self._ommatidia:
            # Add random points near edges
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * 0.95
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            points.append((px, py))

        return np.array(points, dtype=np.float32)  # (721, 2)

    def _render_eye(self, cam_id):
        """Render a single eye camera and return grayscale image."""
        self._renderer.update_scene(self.data, camera=cam_id)
        rgb = self._renderer.render()  # (H, W, 3) uint8
        # Convert to grayscale (luminance)
        gray = np.mean(rgb, axis=2).astype(np.float32) / 255.0  # (H, W), [0, 1]
        return gray

    def _sample_ommatidia(self, gray_image):
        """
        Sample ommatidium values from grayscale image.

        Each ommatidium gets 2 values:
          ch0: bilinear sampled brightness at grid point
          ch1: brightness at a slightly offset point (simulates R7/R8)

        Returns (721, 2) float32 in [0, 1].
        """
        H, W = gray_image.shape
        result = np.zeros((self._ommatidia, 2), dtype=np.float32)

        for i, (px, py) in enumerate(self._grid):
            # Bilinear sample for channel 0
            x0 = int(np.floor(px))
            y0 = int(np.floor(py))
            x1 = min(x0 + 1, W - 1)
            y1 = min(y0 + 1, H - 1)
            x0 = max(x0, 0)
            y0 = max(y0, 0)

            fx = px - x0
            fy = py - y0

            ch0 = (
                gray_image[y0, x0] * (1 - fx) * (1 - fy) +
                gray_image[y0, x1] * fx * (1 - fy) +
                gray_image[y1, x0] * (1 - fx) * fy +
                gray_image[y1, x1] * fx * fy
            )

            # Channel 1: slightly offset (2px right) for photoreceptor diversity
            px2 = min(px + 2.0, W - 1)
            x0b = int(np.floor(px2))
            x1b = min(x0b + 1, W - 1)
            x0b = max(x0b, 0)
            fxb = px2 - x0b

            ch1 = (
                gray_image[y0, x0b] * (1 - fxb) * (1 - fy) +
                gray_image[y0, x1b] * fxb * (1 - fy) +
                gray_image[y1, x0b] * (1 - fxb) * fy +
                gray_image[y1, x1b] * fxb * fy
            )

            result[i, 0] = ch0
            result[i, 1] = ch1

        return result

    def process(self):
        """
        Render both eyes and return ommatidium array.

        Returns:
            vision_obs: np.ndarray of shape (2, 721, 2), dtype=float32, values in [0, 1]
              [0, :, :] = left eye ommatidia
              [1, :, :] = right eye ommatidia
        """
        # Render left eye
        gray_left = self._render_eye(self._cam_left_id)
        omm_left = self._sample_ommatidia(gray_left)

        # Render right eye
        gray_right = self._render_eye(self._cam_right_id)
        omm_right = self._sample_ommatidia(gray_right)

        # Stack: (2, 721, 2)
        vision_obs = np.stack([omm_left, omm_right], axis=0).astype(np.float32)

        # Apply contrast gain to suppress false T2 activation from
        # static backgrounds. gain=0.5 → dimmer areas pulled toward
        # mean; dark looming objects retain contrast, floor fades.
        if self.contrast_gain < 1.0:
            mean_val = vision_obs.mean()
            vision_obs = mean_val + self.contrast_gain * (vision_obs - mean_val)

        return vision_obs
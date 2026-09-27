"""
Q1LiteVisualBridge: Q1 Lite eye cameras → fly ommatidium format.

Same pipeline as the Go2 bridge (unitree_bridge/go2_vision.py): renders
the forward-facing eye_left/eye_right cameras (defined on the base body
in q1lite.xml, looking along the body +Y forward axis with 90 deg FOV)
and grid-samples them into the (2, 721, 2) ommatidium arrays consumed by
visual_system.py (T2 → LC4/LPLC2 → Giant Fiber escape).
"""

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

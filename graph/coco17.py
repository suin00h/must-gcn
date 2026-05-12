"""COCO-17 keypoint graph for skeleton-based action recognition.

Joints (0-indexed):
    0: nose             1: l_eye           2: r_eye
    3: l_ear            4: r_ear
    5: l_shoulder       6: r_shoulder      7: l_elbow         8: r_elbow
    9: l_wrist         10: r_wrist
   11: l_hip           12: r_hip          13: l_knee         14: r_knee
   15: l_ankle         16: r_ankle

Skeleton tree rooted at the nose (joint 0).  Edges follow the canonical PYSKL
COCO-17 inward list, re-rooted so every limb chain points back toward the head.
"""

from . import tools


NUM_NODE = 17

# (child, parent) — child points back toward the root (nose).
INWARD = [
    (1, 0), (2, 0), (3, 1), (4, 2),          # face: eyes -> nose, ears -> eyes
    (5, 0), (6, 0),                          # shoulders -> nose (proxy for neck)
    (7, 5), (9, 7),                          # left arm
    (8, 6), (10, 8),                         # right arm
    (11, 5), (12, 6),                        # hips -> shoulders
    (13, 11), (15, 13),                      # left leg
    (14, 12), (16, 14),                      # right leg
]
OUTWARD = [(j, i) for (i, j) in INWARD]
SELF_LINK = [(i, i) for i in range(NUM_NODE)]
NEIGHBOUR_LINK = INWARD + OUTWARD


class Graph:
    """3-partition spatial graph (identity, inward, outward) → A: (3, V, V)."""

    def __init__(self, labeling_mode: str = 'spatial'):
        self.num_node = NUM_NODE
        self.self_link = SELF_LINK
        self.inward = INWARD
        self.outward = OUTWARD
        self.A = self._build(labeling_mode)

    def _build(self, labeling_mode):
        if labeling_mode == 'spatial':
            return tools.get_spatial_graph(NUM_NODE, SELF_LINK, INWARD, OUTWARD)
        raise ValueError(f'unknown labeling_mode: {labeling_mode}')


if __name__ == '__main__':
    g = Graph()
    print('A shape:', g.A.shape)            # (3, 17, 17)
    print('A.sum() per partition:',
          [float(g.A[i].sum()) for i in range(3)])  # I=17, In/Out: edge counts after normalisation

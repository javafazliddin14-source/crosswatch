"""Static layout of the (single, fixed) camera view.

Hand-measured geometry lives here as code (normalised x, y in [0, 1]); the
data-driven part - where vehicles drive and in which direction - is learned
from the sample videos by `tools/build_scene.py` and stored in
weights/scene_flow.npz. Rules query both through the `SCENE` singleton.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src import config as C

REF_W, REF_H = 1920.0, 1080.0   # geometry below was measured on a 1920x1080 view


def _n(pts) -> np.ndarray:
    return np.array([(x / REF_W, y / REF_H) for x, y in pts], np.float32)


# Zebra crossings, two versions:
#  * STRIPES: the painted crossing itself - "a pedestrian is on the crossing" (failure_to_yield);
#  * CROSSWALKS: stripes plus the band people actually walk in (on the samples most use a strip
#    ~1-2 m beside the stripes) - walking there is not jaywalking.
STRIPES = {
    "north": _n([(320, 582), (1705, 462), (1718, 508), (362, 662)]),
    "southwest": _n([(168, 732), (352, 706), (915, 1080), (582, 1080)]),
}
CROSSWALKS = {
    "north": _n([(318, 560), (1720, 438), (1840, 470), (1840, 540), (355, 705)]),   # across both carriageways
    "southwest": _n([(150, 732), (385, 700), (990, 1080), (570, 1080)]),
}
# Raised areas inside the asphalt where pedestrians legitimately stand: the
# central median, the crossing refuge and the three brick traffic islands,
# plus the left plaza. The learned road mask bleeds over them (vehicle boxes
# are wide), so they are carved out explicitly.
NON_ROAD = [
    _n([(90, 112), (130, 106), (1200, 496), (1160, 524)]),                 # central median
    _n([(1170, 530), (1315, 536), (1305, 576), (1175, 570)]),             # crossing refuge
    _n([(500, 775), (628, 655), (760, 742)]),                              # island, upper
    _n([(105, 940), (330, 840), (430, 920)]),                              # island, lower left
    _n([(670, 852), (872, 802), (962, 882), (722, 900)]),                 # island, lower right
    _n([(0, 515), (335, 565), (340, 690), (160, 740), (0, 810)]),         # left plaza / sidewalk
]
# Curbside lane of the far carriageway in front of the buildings: cars park and
# load there for minutes (one stays for a whole sample clip). Parked, not "stopped".
CURB_PARKING = [_n([(330, 150), (560, 140), (1150, 320), (1080, 355), (700, 285), (330, 200)])]
# Stop line of the approach that faces the camera (vehicles cross it downwards).
STOP_LINE = _n([(300, 527), (935, 458)])
# Signal heads facing the camera: (x1, y1, x2, y2), measured on the 4K frames.
SIGNAL_HEADS = [
    (2290 / 3840, 715 / 2160, 2350 / 3840, 835 / 2160),   # vehicle head, median pole
    (515 / 3840, 960 / 2160, 585 / 3840, 1045 / 2160),    # pedestrian head, left pole
]

# One zone per direction of travel, for congestion: (polygon, min vehicles, signal-controlled?)
TRAFFIC_ZONES = {
    "southeast_exit": (_n([(760, 560), (1920, 470), (1920, 1080), (1150, 1080)]), 8, False),
    # whole near carriageway, from the top of the frame down to the stop line (left of the median)
    "near_approach": (_n([(60, 95), (135, 100), (1000, 460), (300, 530), (150, 330)]), 8, True),
    "far_carriageway": (_n([(560, 60), (900, 60), (1750, 470), (1150, 500)]), 5, False),
}

GRID_W, GRID_H = 96, 54         # resolution of the learned flow field / road mask
N_DIR = 8                       # direction bins (45 degrees each)


@dataclass
class Scene:
    crosswalks: dict = field(default_factory=lambda: CROSSWALKS)
    stop_line: np.ndarray = field(default_factory=lambda: STOP_LINE)
    signal_heads: list = field(default_factory=lambda: SIGNAL_HEADS)
    traffic_zones: dict = field(default_factory=lambda: TRAFFIC_ZONES)
    # learned (None until loaded)
    flow_hist: np.ndarray | None = None    # (GRID_H, GRID_W, N_DIR) normalised direction histograms
    flow_count: np.ndarray | None = None   # (GRID_H, GRID_W) number of moving-vehicle samples
    road_mask: np.ndarray | None = None    # (GRID_H, GRID_W) bool: cells where vehicles drive
    road_core: np.ndarray | None = None    # road_mask eroded by one cell: clearly on the carriageway

    # ---- geometry ---------------------------------------------------------------
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return (min(GRID_H - 1, max(0, int(y * GRID_H))), min(GRID_W - 1, max(0, int(x * GRID_W))))

    def in_crosswalk(self, x: float, y: float, strict: bool = False) -> str | None:
        """Name of the crossing at (x, y); strict=True uses the painted stripes only."""
        for name, poly in (STRIPES if strict else self.crosswalks).items():
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return name
        return None

    def zone_of(self, x: float, y: float) -> str | None:
        for name, (poly, _, _) in self.traffic_zones.items():
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return name
        return None

    def in_parking(self, x: float, y: float) -> bool:
        return any(cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in CURB_PARKING)

    def on_road(self, x: float, y: float) -> bool:
        """On the carriageway proper: inside the (eroded) learned mask and off every raised island."""
        if self.road_core is None:
            return False
        if not self.road_core[self.cell(x, y)]:
            return False
        return all(cv2.pointPolygonTest(p, (float(x), float(y)), False) < 0 for p in NON_ROAD)

    def stop_line_side(self, x: float, y: float) -> float:
        """> 0 beyond (below) the stop line, < 0 before it, in normalised y units."""
        (x1, y1), (x2, y2) = self.stop_line
        y_line = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        return y - y_line

    def near_stop_line_span(self, x: float, margin: float = 0.02) -> bool:
        (x1, _), (x2, _) = self.stop_line
        return x1 - margin <= x <= x2 + margin

    # ---- learned flow -------------------------------------------------------------
    def direction_prob(self, x: float, y: float, vx: float, vy: float) -> tuple[float, int]:
        """(probability of this heading in this cell, support count)."""
        if self.flow_hist is None:
            return 1.0, 0
        r, c = self.cell(x, y)
        b = direction_bin(vx, vy)
        return float(self.flow_hist[r, c, b]), int(self.flow_count[r, c])

    def dominant_heading(self, x: float, y: float, min_share: float = 0.6, min_count: int = 30):
        """Unit vector of the cell's dominant heading if one direction clearly dominates, else None."""
        if self.flow_hist is None:
            return None
        r, c = self.cell(x, y)
        if self.flow_count[r, c] < min_count:
            return None
        b = int(np.argmax(self.flow_hist[r, c]))
        if self.flow_hist[r, c, b] < min_share:
            return None
        ang = b * 2 * np.pi / N_DIR - np.pi
        return float(np.cos(ang)), float(np.sin(ang))

    def load_flow(self, path: Path) -> "Scene":
        if path.exists():
            d = np.load(path)
            self.flow_hist, self.flow_count, self.road_mask = d["hist"], d["count"], d["road"].astype(bool)
            self.road_core = cv2.erode(self.road_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        return self


def direction_bin(vx: float, vy: float) -> int:
    ang = np.arctan2(vy, vx)  # image coords: y down
    return int(((ang + np.pi) / (2 * np.pi) * N_DIR + 0.5) % N_DIR)


SCENE_FLOW_PATH = C.WEIGHTS_DIR / "scene_flow.npz"
SCENE = Scene().load_flow(SCENE_FLOW_PATH)

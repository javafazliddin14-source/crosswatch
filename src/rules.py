"""Event rules: trajectories + scene layout + signal state -> [start, end] segments per class.

Every rule is a pure function of an `EventContext`. Thresholds are in
normalised image units (x, y in [0, 1]) or in "own box heights per second"
for speeds, which makes them roughly perspective-invariant.

Classes are only emitted where a rule can be precise: under macro-F1 a class
that we predict but that never occurs in the test set costs a full class
worth of score, so noisy detectors are worse than silence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import config as C
from src.scene import SCENE, Scene
from src.segments import clean, flags_to_segments, merge_segments
from src.signals import RED
from src.tracks import Track

DT = 1.0 / C.SAMPLE_FPS
STILL = 0.12          # heights / s below which a vehicle counts as stationary
MOVING = 0.5          # heights / s above which it is clearly moving


@dataclass
class EventContext:
    tracks: dict[int, Track]
    times: np.ndarray          # sampled frame times
    signal: np.ndarray         # per-sample approach-signal state (RED / GREEN / UNKNOWN)
    duration: float
    scene: Scene = field(default_factory=lambda: SCENE)

    def signal_at(self, t: float) -> int:
        i = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        return int(self.signal[i]) if len(self.signal) else -1

    @property
    def vehicles(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.is_vehicle]

    @property
    def people(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.is_person]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index runs [i, j] (inclusive) where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    m = mask.copy()
    for i, j in _runs(~mask):
        if i > 0 and j < len(mask) - 1 and j - i + 1 <= max_gap:
            m[i:j + 1] = True
    return m


# ---------------------------------------------------------------------------------------
# pedestrians
# ---------------------------------------------------------------------------------------
def _reliable_person(p: Track, k: int) -> bool:
    """Big and confident enough that the foot point is trustworthy, and not cut off by the frame edge."""
    return p.h[k] >= 0.035 and p.conf[k] >= 0.45 and p.y[k] < 0.985


def jaywalking(ctx: EventContext) -> list[tuple[float, float]]:
    """A person on the carriageway outside the zebra crossings for >= 1.5 s."""
    segs = []
    for p in ctx.people:
        on = np.array([_reliable_person(p, k) and ctx.scene.on_road(x, y) and ctx.scene.in_crosswalk(x, y) is None
                       for k, (x, y) in enumerate(zip(p.x, p.y))])
        on = _fill_short_gaps(on, 3)
        for i, j in _runs(on):
            dur = p.t[j] - p.t[i]
            moved = np.hypot(p.x[j] - p.x[i], p.y[j] - p.y[i]) / max(float(np.median(p.h[i:j + 1])), 1e-3)
            # >= 1.5 s on the road, walking, actually going somewhere (>= 1 body height);
            # a "person" standing in traffic for over a minute is a static false positive
            if 1.5 <= dur <= 60.0 and moved >= 1.0 and _is_walking_person(p, i, j):
                segs.append((p.t[i], p.t[j] + DT))
    return clean(segs, gap=2.0, min_len=1.5, duration=ctx.duration)


def _is_walking_person(p: Track, i: int, j: int) -> bool:
    """Walking-speed track: rejects riders (a 'person' box on a motorbike moves at vehicle
    speed) and tracker ID jumps between two distant people (a teleport = huge peak speed)."""
    sp = p.speed_rel[i:j + 1]
    return float(np.median(sp)) < 2.5 and float(np.percentile(sp, 90)) < 4.0


def failure_to_yield(ctx: EventContext) -> list[tuple[float, float]]:
    """A vehicle drives through a crossing while a pedestrian is on it, near its path.

    "On it" = on the painted stripes, over the carriageway (not the refuge or the
    kerb), and walking - people waiting at the kerb edge are not on the crossing.
    """
    peds_on: dict[str, list[tuple[float, float, float]]] = {}   # crossing -> (t, x, y)
    for p in ctx.people:
        for k, (t, x, y) in enumerate(zip(p.t, p.x, p.y)):
            if not _reliable_person(p, k) or p.speed_rel[k] < 0.3:
                continue
            cw = ctx.scene.in_crosswalk(x, y, strict=True)
            if cw is not None and ctx.scene.on_road(x, y):
                peds_on.setdefault(cw, []).append((t, x, y))
    ped_arr = {k: np.array(v) for k, v in peds_on.items()}

    segs = []
    for v in ctx.vehicles:
        inside = [ctx.scene.in_crosswalk(x, y) for x, y in zip(v.x, v.y)]
        for name in set(c for c in inside if c):
            mask = np.array([c == name for c in inside]) & (v.speed_rel > MOVING)
            for i, j in _runs(_fill_short_gaps(mask, 2)):
                if name not in ped_arr or v.t[j] - v.t[i] < 0.4:
                    continue
                pa = ped_arr[name]
                conflict = False
                for k in range(i, j + 1):
                    near_t = np.abs(pa[:, 0] - v.t[k]) <= DT / 2
                    if not near_t.any():
                        continue
                    # pedestrian within ~1.5 lane widths of the vehicle along the crossing
                    d = np.hypot(pa[near_t, 1] - v.x[k], (pa[near_t, 2] - v.y[k]) * 0.6)
                    if (d < max(0.06, 1.2 * v.w[k])).any():
                        conflict = True
                        break
                if conflict:
                    segs.append((v.t[i], v.t[j] + DT))
    return clean(segs, gap=1.0, min_len=0.6, duration=ctx.duration)


# ---------------------------------------------------------------------------------------
# signal-controlled approach (the one facing the camera, with the stop line)
# ---------------------------------------------------------------------------------------
def crossing_index(v: Track) -> int | None:
    """First sample where the vehicle's ground point passes the stop line downwards."""
    side = np.array([SCENE.stop_line_side(x, y) for x, y in zip(v.x, v.y)])
    in_span = np.array([SCENE.near_stop_line_span(x) for x in v.x])
    for k in range(1, len(side)):
        if side[k - 1] < 0 <= side[k] and in_span[k] and v.vy[k] > 0 and v.speed_rel[k] > MOVING:
            # a real passage, not box jitter of a car queued on the line:
            # clearly upstream ~1 s before and clearly beyond within 2 s
            later = side[k:k + int(2 * C.SAMPLE_FPS) + 1]
            if side[max(0, k - 5)] < -0.005 and later.max() > 0.03:
                return k
    return None


def red_light(ctx: EventContext, grace: float = 4.0) -> list[tuple[float, float]]:
    """Front of the vehicle crosses the stop line while the approach signal is red.

    The heads we can read belong to movements that run with this approach; its
    own lamp (seen from behind) lags them by a few seconds - on the samples
    cars cross 1-5 s after our red onset in every cycle. `grace` skips that
    window so only clear violations are reported. The event ends when the vehicle leaves
    the intersection (its track ends, or it reaches the bottom / right exits of the
    view), capped at 8 s.
    """
    segs = []
    for v in ctx.vehicles:
        k = crossing_index(v)
        if k is None:
            continue
        t_cross = v.t[k]
        # red since >= `grace` s and still red 1 s later (not a jump-start into the green)
        if not all(ctx.signal_at(t) == RED for t in (t_cross - grace, t_cross, t_cross + 1.0)):
            continue
        end = v.t[-1] + DT
        for m in range(k, len(v.t)):
            if v.y[m] > 0.85 or v.x[m] > 0.95:
                end = v.t[m]
                break
        segs.append((t_cross, min(end, t_cross + 8.0)))
    return clean(segs, gap=0.0, min_len=0.8, duration=ctx.duration)


def stop_line(ctx: EventContext) -> list[tuple[float, float]]:
    """Vehicle stops past the stop line on red without entering the intersection.

    Stopped with its ground point between the stop line and the far edge of the
    north crossing; the event lasts until the signal turns green.
    """
    segs = []
    for v in ctx.vehicles:
        for k in range(len(v.t)):
            s = ctx.scene.stop_line_side(v.x[k], v.y[k])
            if not (0.004 < s < 0.09 and ctx.scene.near_stop_line_span(v.x[k], 0.0)):
                continue
            if v.speed_rel[k] > STILL or ctx.signal_at(v.t[k]) != RED:
                continue
            # the stop itself began on red (a car stuck here since green is congestion, not this)
            i0 = k
            while i0 > 0 and v.speed_rel[i0 - 1] <= STILL:
                i0 -= 1
            if ctx.signal_at(v.t[i0]) != RED or i0 == 0:
                break
            # held still here for >= 2 s
            j = k
            while j + 1 < len(v.t) and v.speed_rel[j + 1] <= STILL:
                j += 1
            if v.t[j] - v.t[k] < 2.0:
                continue
            t_green = v.t[k]
            while t_green < ctx.duration and ctx.signal_at(t_green) == RED:
                t_green += DT
            segs.append((v.t[k], t_green))
            break
    return clean(segs, gap=1.0, min_len=2.0, duration=ctx.duration)


# ---------------------------------------------------------------------------------------
# vehicles
# ---------------------------------------------------------------------------------------
def _in_signal_queue(ctx: EventContext, v: Track, i: int, j: int) -> bool:
    """Stopped on the signal-controlled approach (upstream of the stop line) while it was red."""
    if ctx.scene.zone_of(v.x[i], v.y[i]) != "near_approach":
        return False
    reds = [ctx.signal_at(t) == RED for t in np.linspace(v.t[i], v.t[j], 8)]
    return float(np.mean(reds)) > 0.3


def _fully_visible(v: Track, i: int, margin: float = 0.012) -> bool:
    x1, y1, x2, y2 = v.box(i)
    return x1 > margin and y1 > margin and x2 < 1 - margin and y2 < 1 - margin


def stopped_vehicle(ctx: EventContext, min_stop: float = 10.0,
                    jams: dict[str, list[tuple[float, float]]] | None = None) -> list[tuple[float, float]]:
    """Still >= 10 s on the carriageway, not queued at the signal, not part of a jam."""
    jams = congestion_by_zone(ctx) if jams is None else jams
    segs = []
    for v in ctx.vehicles:
        still = _fill_short_gaps(v.speed_rel < STILL, 3)
        for i, j in _runs(still):
            if v.t[j] - v.t[i] < min_stop:
                continue
            if not (ctx.scene.on_road(v.x[i], v.y[i]) and _fully_visible(v, i) and v.h[i] >= 0.035):
                continue          # off the carriageway, cut by the frame edge, or too far to judge
            if ctx.scene.in_parking(v.x[i], v.y[i]):
                continue          # parked in the curbside lane
            if _in_signal_queue(ctx, v, i, j) or _queued_behind(ctx, v, i, j):
                continue
            zone = ctx.scene.zone_of(v.x[i], v.y[i])
            mid = (v.t[i] + v.t[j]) / 2
            if zone and any(s <= mid <= e for s, e in jams.get(zone, [])):
                continue          # one of many cars in a jam: that is congestion
            segs.append((v.t[i], v.t[j] + DT))
    return clean(segs, gap=2.0, min_len=min_stop, duration=ctx.duration)


def _queued_behind(ctx: EventContext, v: Track, i: int, j: int) -> bool:
    """Another vehicle is standing still close to v in mid-stop: v is part of a queue, not stopped alone."""
    tm = (v.t[i] + v.t[j]) / 2
    k = v.at(tm)
    if k is None:
        return False
    stopped_near = 0
    for o in ctx.vehicles:
        if o.tid == v.tid:
            continue
        m = o.at(tm)
        if m is None or o.speed_rel[m] > STILL:
            continue
        dx, dy = o.x[m] - v.x[k], o.y[m] - v.y[k]
        if np.hypot(dx, dy) < 3.0 * max(v.w[k], v.h[k]):
            stopped_near += 1
    return stopped_near >= 1


def wrong_way(ctx: EventContext) -> list[tuple[float, float]]:
    """Driving against a lane: heading opposite (>= 135 deg) to a spot's strongly dominant
    flow (>= 60 % of the learned headings there), barely ever seen there, for >= 3 s.

    Junction cells carry several legitimate headings (turns), so they never have a
    dominant flow and never fire; only one-way lane cells can.
    """
    if ctx.scene.flow_hist is None:
        return []
    segs = []
    for v in ctx.vehicles:
        bad = np.zeros(len(v.t), bool)
        for k in range(len(v.t)):
            if v.speed_rel[k] < 0.8:
                continue
            dom = ctx.scene.dominant_heading(v.x[k], v.y[k])
            if dom is None:
                continue
            p, _ = ctx.scene.direction_prob(v.x[k], v.y[k], v.vx[k], v.vy[k])
            cosang = (v.vx[k] * dom[0] + v.vy[k] * dom[1]) / max(np.hypot(v.vx[k], v.vy[k]), 1e-9)
            bad[k] = p < 0.03 and cosang < -0.7
        bad = _fill_short_gaps(bad, 3)
        for i, j in _runs(bad):
            dist = np.hypot(v.x[j] - v.x[i], v.y[j] - v.y[i])
            if v.t[j] - v.t[i] >= 3.0 and dist > 0.05:
                end = v.t[-1] + DT if j >= len(v.t) - 3 else v.t[j] + DT
                segs.append((v.t[i], end))
    return clean(segs, gap=2.0, min_len=2.0, duration=ctx.duration)


def congestion(ctx: EventContext) -> list[tuple[float, float]]:
    """A whole direction at a standstill (union over the traffic zones)."""
    segs = [s for zone_segs in congestion_by_zone(ctx).values() for s in zone_segs]
    return clean(segs, gap=3.0, min_len=10.0, duration=ctx.duration)


def congestion_by_zone(ctx: EventContext, min_len: float = 10.0) -> dict[str, list[tuple[float, float]]]:
    """Per traffic zone: periods where it holds many vehicles, most of them still.

    For the signal-controlled approach, a red-phase queue is normal: only a
    standstill during green (the queue does not discharge) counts.
    """
    n = len(ctx.times)
    if n == 0:
        return {}
    idx = {round(float(t), 3): i for i, t in enumerate(ctx.times)}
    zones = list(ctx.scene.traffic_zones)
    count = np.zeros((len(zones), n))
    still = np.zeros((len(zones), n))
    for v in ctx.vehicles:
        for t, x, y, s in zip(v.t, v.x, v.y, v.speed_rel):
            z = ctx.scene.zone_of(x, y)
            i = idx.get(round(float(t), 3))
            if z is None or i is None:
                continue
            zi = zones.index(z)
            count[zi, i] += 1
            still[zi, i] += s < STILL * 1.5
    win = int(3 * C.SAMPLE_FPS)
    kernel = np.ones(win) / win
    out = {}
    for zi, name in enumerate(zones):
        _, min_count, signalled = ctx.scene.traffic_zones[name]
        c = np.convolve(count[zi], kernel, mode="same")
        f = np.convolve(still[zi], kernel, mode="same") / np.maximum(c, 1e-6)
        jam = (c >= min_count) & (f >= 0.7)
        if signalled:
            jam &= ctx.signal != RED
        segs = flags_to_segments(ctx.times, _fill_short_gaps(jam, int(3 * C.SAMPLE_FPS)), DT)
        out[name] = clean(segs, gap=3.0, min_len=min_len, duration=ctx.duration)
    return out


def road_obstacle(ctx: EventContext) -> list[tuple[float, float]]:
    """An animal on the carriageway for >= 2 s (debris is not detectable with COCO classes)."""
    segs = []
    for a in ctx.tracks.values():
        if a.cls not in C.ANIMALS or float(np.mean(a.conf)) < 0.35:
            continue
        on = np.array([ctx.scene.on_road(x, y) for x, y in zip(a.x, a.y)])
        for i, j in _runs(_fill_short_gaps(on, 3)):
            if a.t[j] - a.t[i] >= 2.0:
                segs.append((a.t[i], a.t[j] + DT))
    return clean(segs, gap=3.0, min_len=2.0, duration=ctx.duration)


RULES = {
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "red_light": red_light,
    "stop_line": stop_line,
    "stopped_vehicle": stopped_vehicle,
    "wrong_way": wrong_way,
    "congestion": congestion,
    "road_obstacle": road_obstacle,
}


def run_rules(ctx: EventContext, enabled: tuple[str, ...] | None = None) -> list[list]:
    events = []
    for label, fn in RULES.items():
        if enabled is not None and label not in enabled:
            continue
        for s, e in merge_segments(fn(ctx), gap=0.0):
            events.append([float(s), float(e), label])
    events.sort()
    return events

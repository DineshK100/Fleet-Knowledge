"""VLM event description: prompt, image encoding, JSON parsing and a template fallback."""

import json
from typing import List, Optional, Sequence

import numpy as np

from fleet_memory.llm_client import ChatClient, LLMError, extract_json

HAZARD_TYPES = ('pedestrian_crossing', 'lane_closure', 'road_damage', 'stalled_vehicle', 'other')
LANES = ('left', 'right', 'center', 'all', 'unknown')

_TYPE_SYNONYMS = {
    'pedestrian': 'pedestrian_crossing', 'pedestrian_crossing': 'pedestrian_crossing',
    'jaywalker': 'pedestrian_crossing', 'person': 'pedestrian_crossing', 'walker': 'pedestrian_crossing',
    'lane_closure': 'lane_closure', 'closed_lane': 'lane_closure', 'construction': 'lane_closure',
    'roadwork': 'lane_closure', 'road_work': 'lane_closure', 'cones': 'lane_closure',
    'traffic_cones': 'lane_closure', 'blocked_lane': 'lane_closure', 'obstruction': 'lane_closure',
    'road_damage': 'road_damage', 'pothole': 'road_damage', 'debris': 'road_damage',
    'damaged_road': 'road_damage', 'bump': 'road_damage', 'road_debris': 'road_damage',
    'stalled_vehicle': 'stalled_vehicle', 'stopped_vehicle': 'stalled_vehicle',
    'broken_down_vehicle': 'stalled_vehicle', 'disabled_vehicle': 'stalled_vehicle',
    'parked_vehicle': 'stalled_vehicle', 'vehicle': 'stalled_vehicle',
}

SYSTEM_PROMPT = (
    'You are the event writer of an autonomous car that is part of a fleet. The car just had a '
    '"surprise" event (hard braking, near-miss, collision or a violent jolt). You receive a few '
    'front-camera frames in time order around the event plus the vehicle state. Describe the '
    'road hazard that most likely caused the event so that other cars of the fleet can be warned '
    'before reaching the same place. Answer with ONE JSON object only, no prose, with keys:\n'
    '  "hazard_type": one of ' + ', '.join(f'"{t}"' for t in HAZARD_TYPES) + ', where\n'
    '      pedestrian_crossing = a person on or stepping onto the road;\n'
    '      lane_closure = traffic cones, barriers or road works blocking a lane;\n'
    '      road_damage = pothole, cracks or debris in the road surface itself (not cones);\n'
    '      stalled_vehicle = a stopped or broken-down vehicle in the lane;\n'
    '      other = anything else\n'
    '  "description": one short sentence (max 25 words) describing the hazard and its context\n'
    '  "lane": which lane is affected from the ego point of view: "left", "right", "center", '
    '"all" or "unknown"\n'
    '  "severity": integer 1 (minor) to 5 (critical)\n'
    '  "advice": short driving advice for the next cars, e.g. "approach below 20 km/h"\n'
)


def build_user_prompt(ctx: dict) -> str:
    lines = ['Vehicle state at the event:']
    lines.append(f"- trigger: {ctx.get('trigger', 'unknown')} (magnitude {ctx.get('magnitude', 0):.2f})")
    lines.append(f"- speed before event: {ctx.get('speed_kmh', 0):.0f} km/h")
    if ctx.get('obstacle_type', 'none') != 'none':
        lines.append(f"- on-board perception saw a {ctx['obstacle_type']} obstacle "
                     f"{ctx.get('obstacle_distance', 0):.0f} m ahead")
    else:
        lines.append('- on-board perception saw no obstacle in the path')
    if ctx.get('context'):
        lines.append(f"- context: {ctx['context']}")
    lines.append('The images are in time order; the last ones are at / just after the event.')
    lines.append('Return the JSON object now.')
    return '\n'.join(lines)


def encode_jpeg(frame_rgb: np.ndarray, max_width: int = 640, quality: int = 85) -> bytes:
    """Encode an RGB uint8 image as JPEG (OpenCV, falling back to Pillow)."""
    img = np.ascontiguousarray(frame_rgb)
    h, w = img.shape[:2]
    try:
        import cv2
        if w > max_width:
            img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('cv2.imencode failed')
        return buf.tobytes()
    except ImportError:
        import io
        from PIL import Image
        im = Image.fromarray(img)
        if w > max_width:
            im = im.resize((max_width, int(h * max_width / w)))
        out = io.BytesIO()
        im.save(out, format='JPEG', quality=quality)
        return out.getvalue()


def normalize_hazard_type(value: str) -> str:
    v = str(value or '').strip().lower().replace(' ', '_').replace('-', '_')
    if v in HAZARD_TYPES:
        return v
    if v in _TYPE_SYNONYMS:
        return _TYPE_SYNONYMS[v]
    for key, t in _TYPE_SYNONYMS.items():
        if key in v:
            return t
    return 'other'


def normalize_record(obj: Optional[dict], default_severity: int = 3) -> Optional[dict]:
    """Validate / coerce a parsed VLM answer into the fixed schema."""
    if not isinstance(obj, dict):
        return None
    out = {}
    out['hazard_type'] = normalize_hazard_type(obj.get('hazard_type') or obj.get('type'))
    desc = str(obj.get('description') or '').strip().replace('\n', ' ')
    out['description'] = desc[:300]
    lane = str(obj.get('lane') or 'unknown').strip().lower()
    out['lane'] = lane if lane in LANES else 'unknown'
    try:
        sev = int(round(float(obj.get('severity', default_severity))))
    except (TypeError, ValueError):
        sev = default_severity
    out['severity'] = max(1, min(5, sev))
    out['advice'] = str(obj.get('advice') or '').strip()[:200]
    return out


def parse_vlm_reply(text: str, default_severity: int = 3) -> Optional[dict]:
    return normalize_record(extract_json(text), default_severity)


def template_record(ctx: dict, severity: int, with_text: bool = True) -> dict:
    """No-VLM record built from the trigger and perception only (baseline E2c / fallback)."""
    obs = ctx.get('obstacle_type', 'none')
    trig = ctx.get('trigger', '')
    if obs == 'walker':
        htype = 'pedestrian_crossing'
    elif obs == 'static':
        htype = 'lane_closure'
    elif obs == 'vehicle':
        htype = 'stalled_vehicle' if ctx.get('obstacle_stationary', True) else 'other'
    elif trig in ('jerk', 'hard_brake'):
        htype = 'road_damage'
    else:
        htype = 'other'
    rec = {'hazard_type': htype, 'lane': 'unknown', 'severity': severity, 'advice': ''}
    rec['description'] = (f"{htype.replace('_', ' ')} detected after {trig.replace('_', ' ')}"
                          if with_text else '')
    return rec


class VLMDescriber:
    """Calls the VLM and falls back to the template record on any failure."""

    def __init__(self, client: Optional[ChatClient], max_image_width: int = 640, logger=None):
        self.client = client
        self.max_image_width = max_image_width
        self.logger = logger

    def describe(self, frames: Sequence[np.ndarray], ctx: dict, severity: int):
        """Returns (record_dict, source, raw_reply)."""
        if self.client is None:
            return template_record(ctx, severity, with_text=False), 'template', ''
        try:
            images: List[bytes] = [encode_jpeg(f, self.max_image_width) for f in frames]
            reply = self.client.chat(SYSTEM_PROMPT, build_user_prompt(ctx), images)
            rec = parse_vlm_reply(reply, severity)
            if rec is None:
                raise LLMError(f'could not parse reply: {reply[:200]!r}')
            # never let the VLM under-rate a collision
            rec['severity'] = max(rec['severity'], severity if ctx.get('trigger') == 'collision' else 1)
            return rec, 'vlm', reply
        except Exception as exc:
            if self.logger:
                self.logger.warning(f'VLM failed ({exc}); using template record')
            return template_record(ctx, severity, with_text=True), 'template', json.dumps({'error': str(exc)})

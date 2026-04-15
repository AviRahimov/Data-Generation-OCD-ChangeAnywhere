"""Map ADE20K semantic classes and domain object types to inpainting prompts."""

ADE20K_CLASS_TO_BACKGROUND = {
    0: "wall",
    1: "building",
    2: "sky",
    3: "floor",
    4: "tree",
    5: "ceiling",
    6: "road",
    7: "bed",
    8: "windowpane",
    9: "grass",
    10: "cabinet",
    11: "sidewalk",
    12: "person",
    13: "earth",
    14: "door",
    15: "table",
    16: "mountain",
    17: "plant",
    29: "fence",
    34: "rock",
    46: "sand",
    52: "path",
    53: "water",
    91: "dirt track",
    94: "palm",
    123: "field",
    126: "lake",
    128: "soil",
    133: "bush",
    134: "gravel",
}

TERRAIN_BACKGROUND_CLASSES = {9, 13, 46, 52, 91, 123, 128, 134, 6, 11}

TERRAIN_LABELS = {
    9: "green grass field",
    13: "dry brown earth ground",
    46: "sandy terrain",
    52: "narrow dirt path",
    91: "unpaved dirt track",
    123: "open grassy field",
    128: "exposed soil ground",
    134: "gravel surface",
    6: "dirt road",
    11: "sidewalk edge",
}

OBJECT_PROMPTS = {
    "rock": [
        "a clearly visible dark gray rock sitting on {bg}, sharp contrast, aerial drone photograph, high resolution",
        "a distinct rough stone with visible shadow on {bg}, top-down drone view, sharp detail",
        "a prominent medium-sized rock casting a shadow on {bg}, overhead aerial photo, high contrast",
    ],
    "sand_patch": [
        "a clearly visible bright sandy patch contrasting with {bg}, aerial drone photograph, sharp detail",
        "a distinct area of exposed light sand on {bg}, top-down drone view, clear boundary",
    ],
    "bush": [
        "a clearly visible dark green bush contrasting with {bg}, aerial drone photograph, sharp detail",
        "a distinct small scrubby green vegetation on {bg}, overhead drone photo, visible shadow",
    ],
    "pit": [
        "a clearly visible dark shallow pit in {bg}, aerial drone photograph, sharp shadow edges",
        "a distinct depression with dark shadow in {bg}, top-down drone view, high contrast",
    ],
    "debris": [
        "a clearly visible piece of debris contrasting with {bg}, aerial drone photograph, sharp detail",
        "distinct scattered litter objects on {bg}, overhead drone view, visible contrast",
    ],
    "bag": [
        "a clearly visible crumpled white plastic bag on {bg}, aerial drone photograph, high contrast",
    ],
}

DISAPPEAR_PROMPTS = [
    "{bg}, smooth natural terrain, aerial drone photograph, no objects, clean ground",
    "{bg}, undisturbed ground, top-down drone view, high resolution, seamless",
    "{bg}, empty flat terrain, overhead aerial photo, natural lighting",
]

OBJECT_WEIGHTS = {
    "rock": 0.64,
    "sand_patch": 0.10,
    "bush": 0.06,
    "pit": 0.06,
    "debris": 0.08,
    "bag": 0.06,
}

_BROWN_BG = {13, 91, 128, 6, 52}   # earth, dirt track, soil, road, path
_LIGHT_BG = {46, 134, 11}          # sand, gravel, sidewalk
_GREEN_BG = {9, 123}               # grass, field

CONTRAST_WEIGHTS = {
    "brown": {"bush": 0.45, "bag": 0.20, "debris": 0.15, "pit": 0.10, "rock": 0.10},
    "light": {"bush": 0.40, "rock": 0.25, "pit": 0.15, "debris": 0.10, "bag": 0.10},
    "green": {"rock": 0.45, "bag": 0.20, "debris": 0.15, "pit": 0.10, "sand_patch": 0.10},
}


def get_background_label(class_id: int) -> str:
    return TERRAIN_LABELS.get(class_id, "natural terrain ground")


def is_terrain_background(class_id: int) -> bool:
    return class_id in TERRAIN_BACKGROUND_CLASSES


def _bg_tone(bg_class_id: int) -> str:
    if bg_class_id in _GREEN_BG:
        return "green"
    if bg_class_id in _LIGHT_BG:
        return "light"
    return "brown"


def get_appearance_prompt(object_type: str, bg_class_id: int, rng=None) -> str:
    import random
    _rng = rng or random
    bg = get_background_label(bg_class_id)
    templates = OBJECT_PROMPTS.get(object_type, OBJECT_PROMPTS["rock"])
    return _rng.choice(templates).format(bg=bg)


def get_disappearance_prompt(bg_class_id: int, rng=None) -> str:
    import random
    _rng = rng or random
    bg = get_background_label(bg_class_id)
    return _rng.choice(DISAPPEAR_PROMPTS).format(bg=bg)


def sample_object_type(rng=None, bg_class_id=None) -> str:
    """Sample an object type that contrasts with the background."""
    import random
    _rng = rng or random
    if bg_class_id is not None:
        tone = _bg_tone(bg_class_id)
        weights_map = CONTRAST_WEIGHTS.get(tone, OBJECT_WEIGHTS)
    else:
        weights_map = OBJECT_WEIGHTS
    types = list(weights_map.keys())
    weights = [weights_map[t] for t in types]
    return _rng.choices(types, weights=weights, k=1)[0]

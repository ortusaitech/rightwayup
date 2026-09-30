"""RightWayUp: full-circle image roll estimation with calibrated abstention, by ORTUS AI.

    from rightwayup import Orienter
    o = Orienter(tier="max")
    r = o.predict("frame.jpg")          # r.angle_cw, r.confidence, r.abstain
    upright = o.correct("frame.jpg")    # PIL image turned upright (unchanged if the model abstains)
"""
__version__ = "1.0.0"

from .core import TIERS, Orienter, Result, confidence, decode, letterbox, load_image

__all__ = ["Orienter", "Result", "TIERS", "letterbox", "decode", "confidence", "load_image", "__version__"]

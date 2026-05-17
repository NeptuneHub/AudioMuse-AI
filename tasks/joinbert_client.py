"""
Thin wrapper around joinbert/inference.py:Router for instant playlist routing.
Sanitizes user input and provides singleton access to the trained JointBERT model.
"""
import re
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
JOINBERT_DIR = (HERE.parent / "joinbert").resolve()
sys.path.insert(0, str(JOINBERT_DIR))

try:
    from inference import Router
except ImportError:
    Router = None


_router: Optional[Router] = None


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input for JointBERT inference.
    - Strip NUL bytes and control characters
    - Keep only printable chars, newlines, tabs, spaces
    - Truncate to 512 chars
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Remove NUL bytes and control characters (except \n, \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Keep only printable + \n, \t, space
    text = re.sub(r'[^\x20-\x7e\n\t]', '', text)

    # Truncate to 512
    text = text[:512]

    # Strip leading/trailing whitespace
    return text.strip()


def get_router() -> Optional[Router]:
    """
    Lazy-load the JointBERT router singleton.
    Returns None if model files are missing or cannot be loaded.
    """
    global _router
    if _router is None:
        try:
            _router = Router()
        except Exception as e:
            print(f"[joinbert_client] Failed to load Router: {e}")
            return None
    return _router


def route_query(text: str) -> tuple[list, float]:
    """
    Route a user query through JointBERT.

    Args:
        text: User query (will be sanitized)

    Returns:
        (tool_calls, max_confidence) where:
        - tool_calls: list of (tool_name, tool_args) tuples from Router.dispatch()
        - max_confidence: max probability across all intents (0.0 if no intents)
    """
    router = get_router()
    if router is None:
        return [], 0.0

    clean_text = sanitize_user_input(text)
    if not clean_text:
        return [], 0.0

    try:
        intents, entities, intent_probs = router.predict(clean_text)
        max_confidence = float(max(intent_probs)) if len(intent_probs) > 0 else 0.0
        tool_calls = router.dispatch(intents, entities)
        return tool_calls, max_confidence
    except Exception as e:
        print(f"[joinbert_client] JointBERT prediction failed: {e}")
        return [], 0.0

"""Shared builders for the orchestrator tests."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.adapters.outbound.files import load_product
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import Transcript

REPO = Path(__file__).resolve().parents[3]
PRODUCTS = REPO / "config" / "products"


def product(name: str = "live_assist.support") -> ProductConfig:
    """Load one of the real product configs."""
    return load_product(PRODUCTS / f"{name}.yaml")


def transcript(
    lines: Optional[List[Tuple[str, float, float, str]]] = None
) -> Transcript:
    """Return a small two-speaker transcript."""
    lines = lines or [
        ("a", 0.5, 2.0, "Buenos días, ¿en qué puedo ayudarle hoy?"),
        ("c", 2.5, 6.0, "Tengo un problema con mi factura de este mes."),
        ("c", 6.2, 7.0, "Sí."),
        ("a", 7.5, 9.0, "Déjeme revisar su cuenta ahora mismo."),
        ("c", 9.5, 13.0, "Me cobraron dos veces el mismo servicio ayer."),
    ]
    data: Dict[str, Any] = {
        "session": {"id": "unit session", "language": "es",
                    "type": "soporte", "goal": "resolver",
                    "started_at": "2026-11-21T10:00:00+01:00"},
        "speakers": [{"id": "a", "name": "Ana", "enrolled": True},
                     {"id": "c", "name": "Carlos", "enrolled": False}],
        "utterances": [
            {"id": f"t{i}.s0", "turn": i, "index": 0, "speaker": s,
             "start": a, "end": b, "text": t}
            for i, (s, a, b, t) in enumerate(lines)
        ],
    }
    return Transcript.model_validate(data)

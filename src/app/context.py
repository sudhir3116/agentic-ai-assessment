from dataclasses import dataclass, field


@dataclass
class Ctx:
    """Per-request runtime context: tools and the dynamic prompt read the caller and thread from here."""

    session_id: str
    user_id: str
    token: str = field(repr=False)

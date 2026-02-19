from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, NewType

EntityId = NewType("EntityId", str)
SettlementId = NewType("SettlementId", str)
RoadId = NewType("RoadId", str)
BuildingId = NewType("BuildingId", str)
WaterId = NewType("WaterId", str)
VegId = NewType("VegId", str)

LayerKind = Literal["terrain", "water", "vegetation", "settlement", "road", "building"]

Severity = Literal["INFO", "WARN", "ERROR"]


class RuleType(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


@dataclass(frozen=True)
class Violation:
    code: str
    rule_type: RuleType
    severity: Severity
    message: str
    entity_id: str | None = None

from typing import Dict

from dataclasses import dataclass, field

@dataclass
class MatchResult:
    """Classe para armazenar resultados de correspondência entre feature e step."""
    step_file: str
    confidence: float
    match_reasons: Dict[str, float] = field(default_factory=dict)
    
    def add_reason(self, reason: str, score: float):
        """Adiciona uma razão de correspondência com sua pontuação."""
        self.match_reasons[reason] = score
        self.confidence = sum(self.match_reasons.values())
    
    def __lt__(self, other):
        """Permite ordenação por confiança (decrescente)."""
        return self.confidence > other.confidence
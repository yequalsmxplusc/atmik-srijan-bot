"""
Semantic labeling system for events
Automatically assigns semantic intent labels based on event content
"""
import re
from typing import List


class SemanticLabeler:
    """
    Industry-standard semantic classifier using regex patterns.
    Handles implicit categories through multi-signal matching.
    """
    
    INTENT_PATTERNS = {
        "case_study": {
            "strong": [
                r"\bcase[\s-]study\b",
                r"\bace\s+the\s+case\b",
                r"\bcase[\s-]o[\s-]mania\b",
                r"\bentropy\b.*\bcase\b",
            ],
            "context": [
                r"\bbusiness\s+case\b",
                r"\banalyze\s+case\b",
                r"\breal\s+world\s+case\b"
            ],
            "exclude": [
                r"\bcold\s+case\b.*\bmurder\b",
                r"\bcold\s+case\b.*\bdetective\b"
            ]
        },
        "stock_trading": {
            "strong": [
                r"\bstock\b.*\btrad",
                r"\bbeat\s+the\s+market\b",
                r"\bcapital\s+clash\b",
                r"\binvestomania\b",
                r"\bequity\b",
                r"\bportfolio\b.*\bstock"
            ],
            "context": [
                r"\binvest",
                r"\bmarket\b.*\bfinance",
                r"\bmutual\s+fund\b"
            ],
            "exclude": []
        },
        "business_plan": {
            "strong": [
                r"\bbusiness\s+plan\b",
                r"\bb[\s-]plan\b",
                r"\bbiznez\b",
                r"\bpitch\s+deck\b"
            ],
            "context": [
                r"\bentrepreneur",
                r"\bstartup\b",
                r"\bventure\b"
            ],
            "exclude": []
        },
        "coding_competition": {
            "strong": [
                r"\bhackathon\b",
                r"\bhackforge\b",
                r"\bcoding\s+competition\b",
                r"\bcompetitive\s+programming\b",
                r"\bpass\s+the\s+baton\b",
                r"\bh42\b",
                r"\buncode\b",
                r"\bsherlocked\b",
                r"\bopenaimer\b",
                r"\bdata\s+drift\b"
            ],
            "context": [
                r"\bprogramming\b",
                r"\balgorithm\b",
                r"\bcoding\b"
            ],
            "exclude": []
        },
        "robotics": {
            "strong": [
                r"\brobot",
                r"\brobosoccer\b",
                r"\barduino\b",
                r"\bcircuit\s+design\b"
            ],
            "context": [],
            "exclude": []
        },
        "quiz": {
            "strong": [
                r"\bquiz\b",
                r"\bquizotopia\b",
                r"\btrivia\b"
            ],
            "context": [],
            "exclude": []
        },
        "photography": {
            "strong": [
                r"\bphotograph",
                r"\bpixellense\b",
                r"\bcamera\b",
                r"\bphoto\b(?!.*\bboot\b)"
            ],
            "context": [],
            "exclude": []
        },
        "gaming": {
            "strong": [
                r"\besports?\b",
                r"\bgaming\b",
                r"\bbgmi\b",
                r"\bvalorant\b",
                r"\bchess\b",
                r"\bfifa\b",
                r"\beafc\b",
                r"\brocket\s+league\b"
            ],
            "context": [],
            "exclude": []
        },
        "merchandise": {
            "strong": [
                r"\bmerch",
                r"\bt[\s-]?shirt\b",
                r"\bshirt\b",
                r"\bcloth",
                r"\bdelivery\b.*\border\b",
                r"\bphase\s+\d"
            ],
            "context": [
                r"\bbuy\b",
                r"\bpurchase\b",
                r"\bprice\b"
            ],
            "exclude": []
        }
    }
    
    @classmethod
    def label_event(cls, name: str, description: str, tags: str, category: str) -> List[str]:
        """
        Assign semantic labels using regex patterns with word boundaries.
        
        Args:
            name: Event name
            description: Event description
            tags: Event tags
            category: Event category
            
        Returns:
            List of semantic intent labels (e.g., ["case_study", "business_plan"])
        """
        labels = []
        full_text = f"{name} {description} {tags}".lower()
        
        for intent, patterns in cls.INTENT_PATTERNS.items():
            # Check exclusions first
            is_excluded = False
            for exc_pattern in patterns.get("exclude", []):
                if re.search(exc_pattern, full_text, re.IGNORECASE):
                    is_excluded = True
                    break
            
            if is_excluded:
                continue
            
            # Require strong signal
            has_strong = False
            for strong_pattern in patterns["strong"]:
                if re.search(strong_pattern, full_text, re.IGNORECASE):
                    has_strong = True
                    break
            
            if has_strong:
                labels.append(intent)
        
        return labels
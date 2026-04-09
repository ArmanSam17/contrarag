"""
detector.py
Calls Claude to detect agreements and contradictions across sources.
"""

import anthropic
import json
from .prompts import build_contradiction_prompt


class ContradictionDetector:
    """
    Uses Claude to detect agreements and contradictions across multiple sources.
    """

    def __init__(self, client: anthropic.Anthropic) -> None:
        """
        Initialize the detector with an Anthropic client.

        Args:
            client: An authenticated Anthropic client instance.
        """
        self.client = client
        self.model = "claude-sonnet-4-5"

    def detect(
        self,
        query: str,
        source_chunks: dict[str, list[str]],
    ) -> dict:
        """
        Run contradiction detection across all sources for a given query.

        Args:
            query: The user's research question.
            source_chunks: Dict mapping source_id to list of relevant chunk texts.

        Returns:
            A dict with keys: agreements, contradictions, source_summaries, confidence.
            On failure, returns a dict with keys: error, raw.
        """
        response_text = ""
        try:
            prompt = build_contradiction_prompt(query, source_chunks)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text

            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            return {
                "error": f"Claude returned invalid JSON: {str(e)}",
                "raw": response_text,
            }
        except Exception as e:
            return {
                "error": str(e),
                "raw": response_text,
            }

"""
formatting.py
Handles formatting of contradiction detector results for Streamlit display.
"""


def format_results_for_display(
    results: dict,
    source_id_to_name: dict,
) -> dict:
    """
    Replace Source A / Source B style labels with actual document names.

    Args:
        results: The raw dict returned by ContradictionDetector.detect().
        source_id_to_name: Dict mapping source_id to human-readable name.

    Returns:
        A new dict with Source A/B/C labels replaced by real document names.
        Returns the original dict unchanged if it contains an error key.
    """
    if "error" in results:
        return results

    source_ids = list(source_id_to_name.keys())
    label_to_name = {}
    for i, source_id in enumerate(source_ids):
        label = f"Source {chr(65 + i)}"
        label_to_name[label] = source_id_to_name[source_id]

    formatted = {}

    formatted["agreements"] = results.get("agreements", [])

    formatted["contradictions"] = []
    for contradiction in results.get("contradictions", []):
        new_positions = {}
        for label, position in contradiction.get("positions", {}).items():
            name = label_to_name.get(label, label)
            new_positions[name] = position
        formatted["contradictions"].append(
            {
                "topic": contradiction.get("topic", ""),
                "positions": new_positions,
            }
        )

    formatted["source_summaries"] = {}
    for label, summary in results.get("source_summaries", {}).items():
        name = label_to_name.get(label, label)
        formatted["source_summaries"][name] = summary

    formatted["confidence"] = results.get("confidence", "low")

    return formatted


def get_confidence_color(confidence: str) -> str:
    """
    Return a hex color corresponding to the confidence level.

    Args:
        confidence: One of "high", "medium", or "low".

    Returns:
        A hex color string.
    """
    colors = {
        "high": "#2ecc71",
        "medium": "#f39c12",
        "low": "#e74c3c",
    }
    return colors.get(confidence.lower(), "#95a5a6")

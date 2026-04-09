"""
prompts.py
Holds all prompt templates used when calling Claude in ContraRAG.
"""


def build_contradiction_prompt(
    query: str,
    source_chunks: dict[str, list[str]],
) -> str:
    """
    Build the contradiction detection prompt for Claude.

    Args:
        query: The user's research question.
        source_chunks: Dict mapping source_id to list of relevant chunk texts.

    Returns:
        A fully formatted prompt string ready to send to Claude.
    """
    source_labels = {}
    label_list = []
    for i, source_id in enumerate(source_chunks.keys()):
        label = f"Source {chr(65 + i)}"
        source_labels[source_id] = label
        label_list.append(label)

    sources_block = ""
    for source_id, chunks in source_chunks.items():
        label = source_labels[source_id]
        sources_block += f"\n=== {label} ===\n"
        for chunk in chunks:
            sources_block += f"{chunk.strip()}\n\n"

    prompt = f"""You are a research analysis assistant specializing in comparing multiple sources on the same topic. Your job is to identify where sources agree and where they contradict each other.

USER QUERY:
{query}

SOURCES:
{sources_block}

INSTRUCTIONS:
- Analyze all sources in relation to the user query.
- Identify specific points where sources agree with each other.
- Identify specific points where sources contradict or differ from each other.
- For each contradiction, name the topic and state clearly what each source says.
- Quote or closely paraphrase directly from the source text — do not invent positions.
- If a source does not address a particular topic, write "Silent on this topic" for that source.
- Write one sentence summarizing each source's overall position on the query.
- Rate your confidence as high, medium, or low based on how directly the sources address the query.

Return ONLY a valid JSON object with exactly this structure and no other text, preamble, or markdown fences:

{{
  "agreements": ["point 1", "point 2"],
  "contradictions": [
    {{
      "topic": "topic name",
      "positions": {{
        {', '.join(f'"{label}": "what {label} says"' for label in label_list)}
      }}
    }}
  ],
  "source_summaries": {{
    {', '.join(f'"{label}": "one sentence summary"' for label in label_list)}
  }},
  "confidence": "high or medium or low"
}}"""

    return prompt

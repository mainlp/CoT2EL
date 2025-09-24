PROMPT_TEMPLATES = {
    # --- Stage 1 Prompts ---
    "cqa_s1_prompt": """Please read the following question, choose the most appropriate answer from the options provided and start your answer with a single letter
Question: {question}
A. {answerA}
B. {answerB}
C. {answerC}
D. {answerD}
E. {answerE}
Answer:""",

    "siqa_s1_prompt": """Please read the following social scenario and the accompanying question, choose the most appropriate answer from the options provided and start your answer with a single letter
Scenario: {context}
Question: {question}
A. {answerA}
B. {answerB}
C. {answerC}
Answer:""",

    "varierr_s1_prompt": """Please determine whether the following statement is true (entailment), undetermined (neutral), or false (contradiction) given the context below and select ONE of the listed options and start your answer with a single letter.
Context: {premise}
Statement: {hypothesis}
A. Entailment
B. Neutral
C. Contradiction
Answer:""",

    # --- Stage 2 Prompt ---
    "s2_extraction_prompt": """The content of your reasoning process is below:
{reasoning}
Please extract and list all the sentences from the aforementioned reasoning process that support each option separately.""",

    # --- Stage 3 Prompt ---
    "markdown_to_structured_json": """
    Convert the given markdown into a structured JSON where each option has two keys: support and oppose. Each key should map to a list of statements from the markdown that either support or oppose that option.

    EXAMPLE JSON OUTPUT:
    {
      "Option A": {
        "support": ["SentenceA.1","SentenceA.2"],
        "oppose": ["SentenceA.3"]
      },
      "Option B": {
        "support": ["SentenceB.1"],
        "oppose": []
      }
    }
    """
}

def get_prompt(template_key, **kwargs):
    """Formats and returns a prompt based on the given key."""
    if template_key not in PROMPT_TEMPLATES:
        raise ValueError(f"Prompt template with key '{template_key}' not found.")
    
    return PROMPT_TEMPLATES[template_key].format(**kwargs)
CLAIM_EXTRACTION_PROMPTS = {
    "en": """Please breakdown the sentence into independent claims.

Example:
Sentence: \"He was born in London and raised by his mother and father until 11 years old.\"
Claims:
- He was born in London.
- He was raised by his mother and father.
- He was raised by his mother and father until 11 years old.

Sentence: \"{sent}\"
Claims:""",
}

MATCHING_PROMPTS = {
    "en": (
        "Given the fact, identify the corresponding words "
        "in the original sentence that help derive this fact. "
        "Please list all words that are related to the fact, "
        "in the order they appear in the original sentence, "
        "each word separated by comma.\nFact: {claim}\n"
        "Sentence: {sent}\nWords from sentence that helps to "
        "derive the fact, separated by comma: "
    )
}


OPENAI_FACT_CHECK_PROMPTS = {
    "en": (
        """Question: {input}

Determine if all provided information in the following claim is true according to the most recent sources of information.

Claim: {claim}
"""
    ),
}

OPENAI_FACT_CHECK_SUMMARIZE_PROMPT = {
    "en": (
        """Question: {input}

Claim: {claim}

Is the following claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""
    ),
}

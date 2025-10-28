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

"zh": """请将句子拆分成独立的断言，确保每个断言表达单一的事实，细化到最小粒度。

示例：
句子：“他出生在伦敦，并在11岁之前由父母抚养长大。”
断言：
- 他出生在伦敦。
- 他由父母抚养长大。
- 他在11岁之前由父母抚养。
- 他有父亲。
- 他有母亲。

句子：“九江市，简称浔，是中华人民共和国江西省下辖的地级市，位于江西省北部。”
断言：
- 九江市简称浔。
- 九江市在中华人民共和国。
- 九江市在江西省。
- 江西省是中华人民共和国的一个省。
- 九江市是地级市。
- 地级市是中国的一个行政区划级别。
- 九江市位于江西省北部。

句子：“{sent}”
断言：""",

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
    ),
    "zh": (
        "根据事实，找出原句中有助于推导该事实的对应词语。"
        "请按它们在原句中出现的顺序列出所有相关词语，"
        "每个词语用空格分隔。\n"
        "# 示例:\n"
        "# 原句: '周杰伦是中国台湾著名歌手，他在2000年发行了第一张专辑。'\n"
        "# 事实: '周杰伦在2000年发行了第一张专辑。'\n"
        "# 回答: '周杰伦 在 2000 年 发行 了 第一 张 专辑'\n"
        "接下来请你根据："
        "事实: {claim}\n"
        "原句: {sent}\n"
        "列出有助于推导事实的句子词语，用空格分隔："
    ),
}



OPENAI_FACT_CHECK_PROMPTS = {
    "en": (
        """Question: {input}

Determine if all provided information in the following claim is true according to the most recent sources of information.

Claim: {claim}
"""
    ),
    "zh": (
        """问题：{input}

根据最新的信息来源，判断以下声明中的所有信息是否属实。

声明：{claim}
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

    "zh": (
        """问题：{input}

声明：{claim}

以下声明是否属实？

回答：{reply}

将此回答总结为一个词，判断声明是否属实： “True”（属实）、“False”（不属实）或 “Not known”（未知）。
"""
    ),
}
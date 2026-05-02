"""
Dual Annotator Classification Script

This script applies two rule-based classifiers to the same 100 tokens

The two classifiers represent different annotation:

    Annotator A :
        Defaults to OBLIGATION unless the surrounding context contains
        clear, unambiguous keywords associated with another category.

    Annotator B :
        Uses a weighted scoring system that allows softer cues
        (hedging language, inclusive phrasing) to push the decision
        toward RECOMMENDATION or PROBABILITY.
"""

import pandas as pd
import re

# CONFIGURATION

INPUT_FILE = "50samples.xlsx"
OUTPUT_FILE = "dual_annotator_validation.xlsx"


# RULE LEXICONS


PROBABILITY_KEYWORDS = {
    # English
    "probably", "likely", "perhaps", "maybe", "possibly", "presumably",
    "apparently", "seems", "seemed", "appears", "appeared", "supposed",
    "suggests", "suggest", "suggested", "evidently", "presumed",
    # Portuguese
    "provavelmente", "talvez", "possivelmente", "aparentemente",
    "parece", "pareceu", "parecem", "presumivelmente", "supostamente",
    "ao que tudo indica", "deve estar", "deve ser", "deve ter",
    "deve haver",
}

RECOMMENDATION_KEYWORDS = {
    # English
    "advise", "advised", "advice", "recommend", "recommended", "suggestion",
    "suggest", "suggested", "ought", "consider", "tip", "tips", "guidance",
    "best", "ideally", "preferably", "encourage", "encouraged",
    # Portuguese
    "aconselha", "aconselhou", "aconselhamos", "recomenda", "recomendou",
    "sugere", "sugerimos", "ideal", "idealmente", "preferencialmente",
    "dica", "dicas", "orienta", "orientação",
}

OBLIGATION_KEYWORDS = {
    # English
    "law", "laws", "legal", "rule", "rules", "regulation", "regulations",
    "required", "requires", "requirement", "mandatory", "obligation",
    "obliged", "duty", "shall", "policy", "policies", "compliance",
    "comply", "must", "forbidden", "prohibited",
    # Portuguese
    "lei", "leis", "regra", "regras", "regulamento", "regulamentos",
    "obrigatório", "obrigatória", "obrigação", "dever", "deveres",
    "norma", "normas", "exigência", "exigido", "exigida",
    "política", "políticas", "proibido", "proibida",
}


# CLASSIFIERS

def get_context(left, right):
    """Combine left and right context into a single lowercase string."""
    text = f"{str(left)} {str(right)}".lower()
    text = re.sub(r"\s+", " ", text)
    return text


def annotator_1_conservative(left_context, right_context):
    """
    Conservative annotator:
    Defaults to OBLIGATION (O) unless there's strong, clear evidence
    of probability or recommendation.
    """
    context = get_context(left_context, right_context)
    has_obligation = any(kw in context for kw in OBLIGATION_KEYWORDS)
    has_probability = any(kw in context for kw in PROBABILITY_KEYWORDS)
    has_recommendation = any(kw in context for kw in RECOMMENDATION_KEYWORDS)

    if has_probability and not has_obligation and not has_recommendation:
        return "P"
    if has_recommendation and not has_obligation and not has_probability:
        return "R"
    return "O"


def annotator_2_liberal(left_context, right_context):
    """
    Liberal annotator:
    Uses a weighted scoring system to allow softer cues
    to push the decision toward P or R.
    """
    context = get_context(left_context, right_context)

    score_o = 0
    score_r = 0
    score_p = 0

    for kw in OBLIGATION_KEYWORDS:
        if kw in context:
            score_o += 2
    for kw in RECOMMENDATION_KEYWORDS:
        if kw in context:
            score_r += 2
    for kw in PROBABILITY_KEYWORDS:
        if kw in context:
            score_p += 2

    # Soft cues for probability (hedging language)
    hedges = ["if ", "might ", "may ", "could ", "talvez", "se "]
    for h in hedges:
        if h in context:
            score_p += 1

    # Soft cues for recommendation (inclusive phrasing)
    inclusive = [" we should", " we must", "devemos", "precisamos", "deveríamos"]
    for inc in inclusive:
        if inc in context:
            score_r += 1

    if score_o == 0 and score_r == 0 and score_p == 0:
        return "O"

    scores = {"O": score_o, "R": score_r, "P": score_p}
    max_score = max(scores.values())
    winners = [cat for cat, s in scores.items() if s == max_score]

    if "O" in winners:
        return "O"
    return winners[0]

# MAIN

def main():
    df = pd.read_excel(INPUT_FILE)

    df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
    df.columns = ['row_num', 'date_country', 'source',
                  'left_context', 'target_verb', 'right_context', 'Manual']

    df['verb_clean'] = df['target_verb'].astype(str).str.strip().str.lower()

    df['Manual'] = df['Manual'].replace('C', 'R')

    df['Conservative'] = df.apply(
        lambda row: annotator_1_conservative(row['left_context'],
                                              row['right_context']),
        axis=1
    )
    df['Liberal'] = df.apply(
        lambda row: annotator_2_liberal(row['left_context'],
                                         row['right_context']),
        axis=1
    )

    df['Conservative_correct'] = df['Conservative'] == df['Manual']
    df['Liberal_correct'] = df['Liberal'] == df['Manual']

    output_cols = ['verb_clean', 'target_verb', 'left_context', 'right_context',
                   'Manual', 'Conservative', 'Liberal',
                   'Conservative_correct', 'Liberal_correct']
    df[output_cols].to_excel(OUTPUT_FILE, index=False)

    # REPORT
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("Comparing automatic classifiers against 100 manually annotated tokens")
    print("=" * 70)

    x_count = (df['Manual'] == 'X').sum()
    print(f"\nNote: {x_count} tokens were manually labeled 'X' (Other).")
    print("The classifiers do not predict X, so these will count as misses.")

    df_no_x = df[df['Manual'] != 'X']
    cons_acc = df_no_x['Conservative_correct'].mean() * 100
    lib_acc = df_no_x['Liberal_correct'].mean() * 100

    print("\n--- OVERALL ACCURACY (excluding X tokens) ---")
    print(f"  Conservative classifier: {cons_acc:.1f}%  ({df_no_x['Conservative_correct'].sum()}/{len(df_no_x)})")
    print(f"  Liberal classifier:      {lib_acc:.1f}%  ({df_no_x['Liberal_correct'].sum()}/{len(df_no_x)})")

    print("\n--- ACCURACY BY VERB (excluding X) ---")
    for verb in ['must', 'should', 'deve', 'precisa']:
        sub = df_no_x[df_no_x['verb_clean'] == verb]
        if len(sub) == 0:
            continue
        cons = sub['Conservative_correct'].mean() * 100
        lib = sub['Liberal_correct'].mean() * 100
        print(f"  {verb:8s}: Conservative {cons:5.1f}%  |  Liberal {lib:5.1f}%  (n={len(sub)})")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

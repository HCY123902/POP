import re
import regex
import json

import random

def get_completion_after_marker(completion):
    marker = "</think>"
    pos = completion.rfind(marker)
    if pos == -1:
        rest = completion
    else:
        rest = completion[pos + len(marker):]
    return rest

def extract_unique_nonempty_match(pattern, rest):
    matches = re.findall(pattern, rest, flags=re.DOTALL)

    if len(matches) != 1:
        return ""
    assert isinstance(matches[0], str)
    match = matches[0].strip()
    if match in ["", "..."]:
        return ""

    return match


def extract_answer_solver(completion):
    """Extract solution from the model output using <answer> tags, \\boxed{} content, or final answers"""
    rest = get_completion_after_marker(completion)
    
    return extract_unique_nonempty_match(r"<answer>(.*?)</answer>", rest)
    
def extract_answer_proposer(completion):
    """Extract solution from the model output using <answer> tags, \\boxed{} content, or final answers"""

    return extract_answer_solver(completion)





def extract_question(completion: str) -> str | None:
    rest = get_completion_after_marker(completion)

    return extract_unique_nonempty_match(r"<problem>(.*?)</problem>", rest)

def extract_rubric(completion: str) -> tuple[str | None, dict | None]:
    rest = get_completion_after_marker(completion)

    rubric_text = extract_unique_nonempty_match(r"<rubric>(.*?)</rubric>", rest)

    if rubric_text == "":
        return "", None

    try:
        rubric = json.loads(rubric_text)
        for ct in rubric:

            if "name" not in rubric[ct]:
                rubric[ct]["name"] = ct

            expected_keys = set(["name", "ground_truth", "weight", "description"])
            expected_keys_1 = set(["name", "gold", "weight", "description"])
            ct_keys = set(rubric[ct].keys())
            if ct_keys != expected_keys and ct_keys != expected_keys_1:
                raise Exception("{} has an incorrect set of keys: Expected {} or {}; Got: {}".format(ct, expected_keys, expected_keys_1, ct_keys))

            if not isinstance(rubric[ct]["name"], str):
                raise Exception("{} has an invalid name: {}".format(ct, rubric[ct]["name"]))

            if "ground_truth" in rubric[ct] and not isinstance(rubric[ct]["ground_truth"], str):
                raise Exception("{} has an invalid ground_truth: {}".format(ct, rubric[ct]["ground_truth"]))
            
            if "gold" in rubric[ct] and not isinstance(rubric[ct]["gold"], str):
                raise Exception("{} has an invalid gold: {}".format(ct, rubric[ct]["gold"]))

            if not isinstance(rubric[ct]["description"], str):
                raise Exception("{} has an invalid description: {}".format(ct, rubric[ct]["description"]))

            rubric[ct]["weight"] = float(rubric[ct]["weight"])

        return rubric_text, rubric
    except Exception as e:
        print("Error when parsing the rubric:\n{}".format(e))
        return "", None
        


def extract_rubric_score_pairwise(completion: str) -> float | None:
    rest = get_completion_after_marker(completion)

    score = extract_unique_nonempty_match(r"<better_answer>(.*?)</better_answer>", rest)

    score = score.strip()

    if score == "":
        return -1.0
    
    if score.casefold() not in ["a", "b"]:
        return -1.0

    if score.casefold() == "a":
        return 1.0

    return 0.0

def extract_rubric_score_pairwise_anchor(completion: str) -> dict | None:
    rest = get_completion_after_marker(completion)

    score_a = extract_unique_nonempty_match(r"<rating_A>(.*?)</rating_A>", rest)
    score_b = extract_unique_nonempty_match(r"<rating_B>(.*?)</rating_B>", rest)

    score_a = score_a.strip()
    score_b = score_b.strip()

    if score_a == "" or score_b == "":
        return None
    
    try:
        score_a = float(score_a)
        score_b = float(score_b)
        return {
            "a": score_a,
            "b": score_b,
        }
    except Exception as e:
        # print("Encountered exception {} when parsing the score {}".format(e, score))
        return None
    

def extract_rubric_score_from_evaluation(completion: str, rubric_text: str) -> tuple[float | None, dict | None]:
    try:
        rubric = json.loads(rubric_text)
        weights = {rubric[ct]["name"].strip().casefold(): float(rubric[ct]["weight"]) for ct in rubric}
        
    except Exception as e:
        print("Error when parsing the rubric:\n{}".format(e))
        return -1.0, None
    
    total_weight = sum([weights[ct_n] for ct_n in weights])

    rest = get_completion_after_marker(completion)

    evaluation_text = extract_unique_nonempty_match(r"<evaluation>(.*?)</evaluation>", rest)
    if evaluation_text == "":
        return -1.0, None

    try:
        evaluation = json.loads(evaluation_text)

        for ct in evaluation:

            if "name" not in evaluation[ct]:
                evaluation[ct]["name"] = ct

            expected_keys = set(["name", "rating", "thoughts"])
            ct_keys = set(evaluation[ct].keys())
            if ct_keys != expected_keys:
                raise Exception("{} has an incorrect set of keys: Expected {}; Got: {}".format(ct, expected_keys, ct_keys))

            if not isinstance(evaluation[ct]["name"], str):
                raise Exception("{} has an invalid name: {}".format(ct, evaluation[ct]["name"]))

            if not isinstance(evaluation[ct]["thoughts"], str):
                raise Exception("{} has an invalid thoughts: {}".format(ct, evaluation[ct]["thoughts"]))
            
            evaluation[ct]["rating"] = float(evaluation[ct]["rating"])
            

        # Deduplicate the criteria
        new_evaluation = {}
        for ct in evaluation:
            ct_name = evaluation[ct]["name"].strip().casefold()
            new_evaluation[ct_name] = {
                "rating": evaluation[ct]["rating"],
                "thoughts": evaluation[ct]["thoughts"]
            }

        score = 0.0
        for ct_name in new_evaluation:
            if ct_name not in weights:
                print("Criterion {} not found among the existing ones {}. Ignoring it.".format(
                    ct_name,
                    weights.keys()
                ))
                continue
            ct_rating = float(new_evaluation[ct_name]["rating"])
            ct_rating = float(max(min(ct_rating, 2), 0))
            score = score + weights[ct_name] * ct_rating
        score = score / total_weight
        # print("Rubric: {}\n\nCompletion: {}\n".format(rubric_text, completion))
        print("Parsing succeeded with final score: {}\n\n".format(score))
        return score, evaluation
    except Exception as e:
        # print("Rubric: {}\n\nCompletion: {}\n".format(rubric_text, completion))
        print("Error when parsing the evaluation:\n{}\n\n".format(e))
        return -1.0, None

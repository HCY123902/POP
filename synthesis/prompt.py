# ========================================
# Question Synthesis Prompt
# ========================================


PROPOSER_PROMPT_WITH_KNOWLEDGE = """You are a problem proposer. Given a piece of knowledge, create a challenging, well-structured, and unambiguous problem.

## Guidelines

* The problem should be assessable.

## Format

* Enclose the problem statement within <problem>...</problem> tags.
  1. In the problem statement, provide the necessary context so that the problem is unambiguous, self-contained, and standalone. Never use: "according to the knowledge", "in the knowledge", "as mentioned", "the knowledge states", "based on the knowledge", etc.
  2. While you should include the necessary context, you should not mention or include any part of the ground truth answer in the problem statement.

* Provide a reference answer within <answer>...</answer> tags. The reference answer should exclude any raw thinking traces."""


PROPOSER_PROMPT_WITH_KNOWLEDGE_CREATIVE_WRITING = """You are a problem proposer. Given a text sample from the description or main body of a book, create a challenging, well-structured, and unambiguous creative writing question.

## Guidelines

* The question should be assessable.
* The question should build upon the given text sample. In particular, to create the question, follow these steps
  1. Extract the key entities, relations, and settings mentioned in the text sample.
  2. Create a creative writing question (write a story, a scene or dialogue in a story, a narrative; expand or continue a story; construct or expand the setting of a story, etc.) that builds upon the text sample and the mentioned entities, relations, and settings.

## Format

* Enclose the question statement within <problem>...</problem> tags.
  1. In the question statement, provide the necessary context so that the question is unambiguous, self-contained, and standalone. Never use: "according to the text sample", "in the text sample", "as mentioned", "the text sample states", "based on the text sample", etc.
  2. While you should include the necessary context, you should not mention or include any reference answer in the question statement.
  3. End your question statement with the length requirement "Length: 1000 words.".

* Provide a reference answer within <answer>...</answer> tags. The reference answer should exclude any raw thinking traces."""








PROPOSER_PROMPT_WITH_KNOWLEDGE_HEALTH_CARE = """You are a problem proposer. Given a piece of knowledge, create a challenging, well-structured, and unambiguous problem.

## Guidelines

* The problem should be assessable.
* The problem should be based on the knowledge and the answer should be derivable from the knowledge. In particular, to create the problem, follow these steps
  1. Extract the list of facts in mentioned in the knowledge.
  2. Create a problem that requires the use of one or more of the extracted facts to answer. Ensure that it is necessary and sufficient to use these facts to derive the correct answer to the problem.
* The problem should ask about a health/medical-related question (pretend that you are a patient and seek health/medical advice; pretend that you are a healthcare professional and ask about professional knowledge, etc.).

## Format

* Enclose the problem statement within <problem>...</problem> tags.
  1. In the problem statement, provide the necessary context so that the problem is unambiguous, self-contained, and standalone. Never use: "according to the knowledge", "in the knowledge", "as mentioned", "the knowledge states", "based on the knowledge", etc.
  2. While you should include the necessary context, you should not mention or include any part of the ground truth answer in the problem statement.

* Provide a reference answer within <answer>...</answer> tags. The reference answer should be final and concise, excluding any raw thinking traces."""





PROPOSER_PROMPT_WITH_KNOWLEDGE_MATH = """You are a problem proposer. Given a piece of knowledge, create a challenging, well-structured, and unambiguous math problem.

## Guidelines

* The problem should be assessable.
* The problem should be based on the knowledge and the answer should be derivable from the knowledge. In particular, to create the problem, follow these steps
  1. Extract the math entities, relations, equations, formulas, facts, questions, or any math objects mentioned in the knowledge.
  2. Create a math problem that requires the use of one or more of the extracted objects to answer. Ensure that it is necessary and sufficient to use these extracted , alone with general reasoning abilities, to derive the correct answer to the problem.
* The math problem should have a single, unique final answer, expressed as a number, a formula, or any other math object. If the problem is a multiple-choice question, the final answer is the correct option identifier.

## Format

* Enclose the problem statement within <problem>...</problem> tags.
  1. In the problem statement, provide the necessary context so that the problem is unambiguous, self-contained, and standalone. Never use: "according to the knowledge", "in the knowledge", "as mentioned", "the knowledge states", "based on the knowledge", etc.
  2. While you should include the necessary context, you should not mention or include any part of the ground truth answer in the problem statement.

* Provide a reference answer within <answer>...</answer> tags. Within <answer>...</answer>, the reference answer should include concise reasoning steps and a final numeric answer or expression enclosed in \\boxed{}."""

PROPOSER_PROMPT_WITH_KNOWLEDGE_ELICITATION_VERIFIABLE = """You are a problem proposer. Given a piece of knowledge, create a challenging, well-structured, and unambiguous problem.

## Guidelines

* The problem should be assessable.
* The problem should be based on the knowledge and the answer should be derivable from the knowledge. In particular, to create the problem, follow these steps
  1. Extract the list of facts in mentioned in the knowledge.
  2. Create a problem that requires the use of one or more of the extracted facts to answer. Ensure that it is necessary and sufficient to use these facts to derive the correct answer to the problem.
* The problem should have a single, unique final answer, in the form of a word, a phrase, a number, or any other short form. If the problem is a multiple-choice question, the final answer is the correct option identifier.

## Format

* Enclose the problem statement within <problem>...</problem> tags.
  1. In the problem statement, provide the necessary context so that the problem is unambiguous, self-contained, and standalone. Never use: "according to the knowledge", "in the knowledge", "as mentioned", "the knowledge states", "based on the knowledge", etc.
  2. While you should include the necessary context, you should not mention or include any part of the ground truth answer in the problem statement.

* Provide a reference answer within <answer>...</answer> tags. Within <answer>...</answer>, the reference answer should include concise thinking steps and a final answer enclosed in \\boxed{}."""






PROPOSER_USER_PROMPT_WITH_KNOWLEDGE = """<knowledge>
{knowledge}
</knowledge>

Now, please create a challenging, well-structured, and unambiguous problem along with its reference answer, using the provided external knowledge as context."""


PROPOSER_USER_PROMPT_WITH_KNOWLEDGE_NO_XML = """Knowledge:
{knowledge}

Now, please create a challenging, well-structured, and unambiguous problem along with its reference answer, using the provided external knowledge as context."""


PROPOSER_USER_PROMPT_WITH_KNOWLEDGE_CREATIVE_WRITING = """<text_sample>
{knowledge}
</text_sample>

Now, please create a challenging, well-structured, and unambiguous creative writing question along with its reference answer, using the provided text sample as context."""














# ========================================
# Question Answering Prompt
# ========================================


SOLVER_SYSTEM_PROMPT = """You are a helpful assistant. Answer the given question.

Think step by step, and enclose your answer in the <answer>...</answer> tags."""

SOLVER_SYSTEM_PROMPT_HEALTH_CARE = """You are an assistant that gives health care and medical advice. Answer the given question.

Think step by step, and enclose your final answer in the <answer>...</answer> tags."""

SOLVER_SYSTEM_PROMPT_CREATIVE_WRITING = """You are a helpful assistant. Answer the given question.

Think step by step, and enclose your answer in the <answer>...</answer> tags."""

SOLVER_SYSTEM_PROMPT_MATH = """You are a helpful assistant. Answer the given question.

Please reason step by step. Enclose your answer in the <answer>...</answer> tags. Within <answer>...</answer>, the answer should include concise reasoning steps and a final numeric answer or expression enclosed in \\boxed{}."""

SOLVER_SYSTEM_PROMPT_ELICITATION_VERIFIABLE = """You are a helpful assistant. Answer the given question.

Think step by step. Enclose your answer in the <answer>...</answer> tags. Within <answer>...</answer>, the answer should include concise thinking steps and a final answer enclosed in \\boxed{}."""

SOLVER_USER_PROMPT = """{question}"""














# ========================================
# Rubric Generation Prompt
# ========================================


VERIFIER_SYSTEM_PROMPT = """Given a piece of knowledge, a problem, its reference answer, and a set of candidate answers, generate a rubric to evaluate the candidate answers. It should consist of 1 to 5 criteria, where each criterion is accompanied by (1) a description, (2) a weight that indicates how important that criterion is to the overall quality, and (3) a gold standard answer if applicable.

## Format

Enclose your final rubric in the <rubric>...</rubric> tags.

Within <rubric>...</rubric>, follow the format:

```
<rubric>
{
    "criterion_1": {
        "name": "(Name of criterion_1)",
        "gold": "(The gold standard answer for criterion_1, if applicable.)",
        "description": "(Description of criterion_1. Mention what details to look for to identify good and bad responses for this criterion.)",
        "weight": (Numeric weight for criterion_1)
    },
    ...,
    "criterion_(N)": {
        "name": "(Name of criterion_(N))",
        "gold": "(The gold standard answer for criterion_(N), if applicable.)",
        "description": "(Description of criterion_(N). Mention what details to look for to identify good and bad responses for this criterion.)",
        "weight": (Numeric weight for criterion_(N))
    }
}
</rubric>
```

## Reminders

* If all the answers are empty, put 'ERROR' as your response.

* Do not overthink. Finalize it once you come up with a reasonable rubric.

* Each criterion needs to distinguish high-quality answers from low-quality ones meaningfully. Take the following process to identify the suitable criteria:
  1. Group answers (both reference and candidate answers) by quality level. In general, the reference answer should have a high quality compared to the candidate answers, but this is not always true.
  2. Identify factors that separate the answers in high-quality groups from those in low-quality groups.
  3. Select the key factors with the highest discriminative power as the criteria to include in the rubric.

* Each criterion needs to be atomic and focus on a single aspect of quality. Different criteria should not overlap with each other.

* Each criterion needs to be specific to the problem as much as possible. If possible, avoid general criteria that are applicable to any problem (i.e., avoid "factuality", "clarity", "conciseness", etc.). If it is necessary to include general criteria, change them to ones that are specific to the problem (e.g, instead of "factuality", put "factuality of XXX").

* For each criterion, whenever applicable, extract or infer the ground truth or gold standard answer from the provided knowledge, and put it in the "gold" field.
  1. The ground truth or gold standard answer needs to be as concise as possible, ideally a single sentence or phrase (e.g., The criterion is about property "YYY" of "XXX". The ground truth answer is "ZZZ". Put "ZZZ" in the "gold" field.).
  2. If not applicable, put "Not applicable" in "gold".

* For each criterion, its description in the "description" field needs to be as specific to the problem as possible.
  1. Do not just give a general description for the criterion. Elaborate on it with key details, key facts, key phrases, keywords, and examples (e.g., The criterion is about understanding of "XXX". To describe it, instead of saying "The level of understanding of XXX", say "The level of understanding of XXX. Correct understanding should include YYY and ZZZ. ...").
  2. Whenever applicable, connect your description to the gold standard answer that you provide in the "gold" field.

* A small number of high-quality criteria are better than a large number of low-quality criteria. 2 specific, highly distinguishing criteria >> 5 general criteria that rate all answers equally."""



VERIFIER_SYSTEM_PROMPT_NO_XML = """Given a piece of knowledge, a problem, its reference answer, and a set of candidate answers, generate a rubric to evaluate the candidate answers. It should consist of 1 to 5 criteria, where each criterion is accompanied by (1) a description, (2) a weight that indicates how important that criterion is to the overall quality, and (3) a ground truth answer if applicable.

## Format

Enclose your final rubric in the <rubric>...</rubric> tags.

Within <rubric>...</rubric>, follow the format:

```
<rubric>
{
    "criterion_1": {
        "name": "(Name of criterion_1)",
        "ground_truth": "(The ground truth answer for criterion_1, if applicable.)",
        "description": "(Description of criterion_1. Mention what details to look for to identify good and bad responses for this criterion.)",
        "weight": (Numeric weight for criterion_1)
    },
    ...,
    "criterion_(N)": {
        "name": "(Name of criterion_(N))",
        "ground_truth": "(The ground truth answer for criterion_(N), if applicable.)",
        "description": "(Description of criterion_(N). Mention what details to look for to identify good and bad responses for this criterion.)",
        "weight": (Numeric weight for criterion_(N))
    }
}
</rubric>
```

## Reminders

* If all the answers are empty, put 'ERROR' as your response.

* Do not overthink. Finalize it once you come up with a reasonable rubric.

* Each criterion needs to distinguish high-quality answers from low-quality ones meaningfully. Take the following process to identify the suitable criteria:
  1. Group answers (both reference and candidate answers) by quality level. In general, the reference answer should have a high quality compared to the candidate answers, but this is not always true.
  2. Identify factors that separate the answers in high-quality groups from those in low-quality groups.
  3. Select the key factors with the highest discriminative power as the criteria to include in the rubric.

* Each criterion needs to be atomic and focus on a single aspect of quality. Different criteria should not overlap with each other.

* Each criterion needs to be specific to the problem as much as possible. If possible, avoid general criteria that are applicable to any problem (i.e., avoid "factuality", "clarity", "conciseness", etc.). If it is necessary to include general criteria, change them to ones that are specific to the problem (e.g, instead of "factuality", put "factuality of XXX").

* For each criterion, whenever applicable, extract or infer the ground truth answer from the provided knowledge, and put it in the "ground_truth" field.
  1. The ground truth answer needs to be as concise as possible, ideally a single sentence or phrase (e.g., The criterion is about property "YYY" of "XXX". The ground truth answer is "ZZZ". Put "ZZZ" in the "ground_truth" field.).
  2. If not applicable, put "Not applicable" in "ground_truth".

* For each criterion, its description in the "description" field needs to be as specific to the problem as possible.
  1. Do not just give a general description for the criterion. Elaborate on it with key details, key facts, key phrases, keywords, and examples (e.g., The criterion is about understanding of "XXX". To describe it, instead of saying "The level of understanding of XXX", say "The level of understanding of XXX. Correct understanding should include YYY and ZZZ. ...").
  2. Whenever applicable, connect your description to the ground truth answer that you provide in the "ground_truth" field.

* A small number of high-quality criteria are better than a large number of low-quality criteria. 2 specific, highly distinguishing criteria >> 5 general criteria that rate all answers equally."""


VERIFIER_USER_PROMPT = """<knowledge>
{knowledge}
</knowledge>

<problem>
{question}
</problem>

<reference_answer>
{ref_answer}
</reference_answer>

<candidate_answers>
{answers}
</candidate_answers>

Now generate the rubric to evaluate the candidate answers."""


VERIFIER_USER_PROMPT_NO_XML = """Knowledge:
{knowledge}

Problem:
{question}

Reference Answer:
{ref_answer}

Candidate Answers:
{answers}

Now generate the rubric to evaluate the candidate answers."""


















# ========================================
# Answer Grading Prompt
# ========================================


VERIFIER_SCORING_SYSTEM_PROMPT = """Given a problem, an evaluation rubric, and a candidate answer, evaluate the candidate answer by giving the answer a rating for each criterion in the rubric.

## Format

Put your rubric-based evaluation in <evaluation>...</evaluation> tags.

```
<evaluation>
{
    "criterion_1": {
        "name": "(Name of criterion_1)",
        "thoughts": "(Your thoughts on how well the answer does for criteron_1. If there is a gold standard answer in the "gold" field, does the answer match the gold standard?)",
        "rating": (Your numeric rating between 0 and 2 according to the description and gold standard answer of criteron_1)
    },
    ...,
    "criterion_(N)": {
        "name": "(Name of criterion_(N))",
        "thoughts": "(Your thoughts on how well the answer does for criteron_(N). If there is a gold standard answer in the "gold" field, does the answer match the gold standard?)",
        "rating": (Your numeric rating between 0 and 2 according to the description and gold standard answer of this criteron_(N))
    }
}
</evaluation>
```

## Reminder

* The rating needs to be either 0 (Bad), 1 (Medium), or 2 (Good). Higher ratings should indicate better qualities.

* For each criterion that has a gold standard answer in the "gold" field, give a rating of 2 only when the response FULLY matches the gold standard. Give a rating of 1 when the response partially matches the gold standard. Give a rating of 0 when the response does not match any part of the gold standard.

* If the candidate answer is empty, put nothing in the <evaluation>...</evaluation> tags."""


VERIFIER_SCORING_SYSTEM_PROMPT_NO_XML = """Given a problem, a candidate answer, and an evaluation rubric, evaluate the candidate answer by giving the answer a rating for each criterion in the rubric.

## Format

Put your rubric-based evaluation in <evaluation>...</evaluation> tags.

```
<evaluation>
{
    "criterion_1": {
        "name": "(Name of criterion_1)",
        "thoughts": "(Your thoughts on how well the answer does for criteron_1. If there is a ground truth, does the answer match the ground truth?)",
        "rating": (Your numeric rating between 0 and 2 according to the description and ground truth of criteron_1)
    },
    ...,
    "criterion_(N)": {
        "name": "(Name of criterion_(N))",
        "thoughts": "(Your thoughts on how well the answer does for criteron_(N). If there is a ground truth, does the answer match the ground truth?)",
        "rating": (Your numeric rating between 0 and 2 according to the description and ground truth of this criteron_(N))
    }
}
</evaluation>
```

## Reminder

* The rating needs to be either 0 (Bad), 1 (Medium), or 2 (Good). Higher ratings should indicate better qualities.

* For each criterion that has a ground truth, give a rating of 2 only when the response FULLY matches the ground truth. Give a rating of 1 when the response partially matches the ground truth. Give a rating of 0 when the response does not match any part of the ground truth.

* If the candidate answer is empty, put nothing in the <evaluation>...</evaluation> tags."""


VERIFIER_SCORING_USER_PROMPT = """<problem>
{question}
</problem>

<rubric>
{rubric}
</rubric>

<candidate_answer>
{answer}
</candidate_answer>

Now evaluate the Candidate Answer."""


VERIFIER_SCORING_USER_PROMPT_NO_XML = """Problem:
{question}

Evaluation Rubric:
{rubric}

Candidate Answer:
{answer}

Now evaluate the Candidate Answer."""












# ========================================
# Pairwise Answer Grading Prompt
# ========================================

VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE = """Given a problem, an evaluation rubric, and two candidate answers, compare the two candidate answers, using the rubric as a reference.

## Format

First, provide your thoughts on how well each answer does according to the rubric, and how they compare with each other. Then, on separate new lines, indicate the better answer using the EXACT tags below.

```
<better_answer>A or B</better_answer>
```

Within the <better_answer>...</better_answer>, put either A or B. If you think candidate answer A is better, put A. If you think candidate answer B is better, put B.

## Reminder

* In the rubric, for each applicable criterion, there is a "gold" field that indicates a gold standard answer for that criterion. Use it when evaluating and comparing the answers whenever possible.

* In the rubric, the "weight" field of each criterion indicates the importance of that criterion. Consider these weights when evaluating and comparing the answers."""

VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE_NO_XML = VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE.replace(
  """there is a "gold" field that indicates a gold standard answer for that criterion""",
  """there is a "ground_truth" field that indicates a ground truth answer for that criterion"""
)


VERIFIER_SCORING_USER_PROMPT_PAIRWISE = """<problem>
{question}
</problem>

<rubric>
{rubric}
</rubric>

<candidate_answer_A>
{answer_a}
</candidate_answer_A>

<candidate_answer_B>
{answer_b}
</candidate_answer_B>

Now compare candidate answer A and candidate answer B and indicate which is better."""

VERIFIER_SCORING_USER_PROMPT_PAIRWISE_NO_XML = """Problem:
{question}

Evaluation Rubric:
{rubric}

Candidate Answer A:
{answer_a}

Candidate Answer B:
{answer_b}

Now compare candidate answer A and candidate answer B and indicate which is better."""



VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE_ANCHOR = """Given a problem, an evaluation rubric, and two candidate answers, compare the two candidate answers and rate each of them, using the rubric as a reference.

## Format

First, provide your thoughts on how well each answer does according to the rubric, and how they compare with each other. Then, on separate new lines, give ratings of each answer using the EXACT tags below.

```
<rating_A>INTEGER_0_TO_10</rating_A>
<rating_B>INTEGER_0_TO_10</rating_B>
```

Rating is on a 0 to 10 scale. Within the tags, put the rating as an integer from 0 to 10. Higher rating means better answer quality. Equal ratings imply a tie.

## Reminder

* In the rubric, for each applicable criterion, there is a "gold" field that indicates a gold standard answer for that criterion. Use it when evaluating and comparing the answers whenever possible.

* In the rubric, the "weight" field of each criterion indicates the importance of that criterion. Consider these weights when evaluating and comparing the answers."""

VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE_ANCHOR_NO_XML = VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE_ANCHOR.replace(
  """there is a "gold" field that indicates a gold standard answer for that criterion""",
  """there is a "ground_truth" field that indicates a ground truth answer for that criterion"""
)

VERIFIER_SCORING_USER_PROMPT_PAIRWISE_ANCHOR = """<problem>
{question}
</problem>

<rubric>
{rubric}
</rubric>

<candidate_answer_A>
{answer_a}
</candidate_answer_A>

<candidate_answer_B>
{answer_b}
</candidate_answer_B>

Now compare candidate answer A and candidate answer B and rate each of them."""

VERIFIER_SCORING_USER_PROMPT_PAIRWISE_ANCHOR_NO_XML = """Problem:
{question}

Evaluation Rubric:
{rubric}

Candidate Answer A:
{answer_a}

Candidate Answer B:
{answer_b}

Now compare candidate answer A and candidate answer B and rate each of them."""



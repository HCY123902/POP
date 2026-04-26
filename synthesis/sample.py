import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


from tqdm import tqdm
import re

import random
import json

from transformers import AutoTokenizer
from argparse import ArgumentParser

from time import strftime, localtime


from knowledge_base import Knowledgebase
import copy

from typing import Union, Any
from datasets import Dataset, concatenate_datasets

from reward_utils import *
from prompt import *

import requests
from tqdm import tqdm
import time

from openai import OpenAI

vllm_client = None
openai_client = None


def generate(llm, msgs, sampling_params, use_peft, peft_dir, vllm_port=8000, client=None):
    if isinstance(llm, str):
        assert client is not None
        response = client.chat.completions.create(
            model=llm,
            messages=msgs,
            n=sampling_params.n,
            max_tokens=sampling_params.max_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
        )
        return [choice.message.content for choice in response.choices]

    else:
        if use_peft:
            o = llm.chat([msgs], sampling_params, lora_request=LoRARequest("adapter", 1, peft_dir), use_tqdm=False)
        else:
            o = llm.chat([msgs], sampling_params, use_tqdm=True)

        return [output.text for output in o[0].outputs]



def openai_generate(model, messages, sampling_params, client=None):
    price_map = {
        "gpt-4.1-mini": {"input": 0.4/(10**6), "output": 1.6/(10**6)},
        "gpt-4o-mini-2024-07-18": {"input": 0.15/(10**6), "output": 0.6/(10**6)},
    }
    assert client is not None
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        n=sampling_params.n,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
    )
    return [choice.message.content for choice in response.choices], price_map[model]["input"] * response.usage.prompt_tokens + price_map[model]["output"] * response.usage.completion_tokens


from typing import Union, Any

def get_prompt(role, tokenizer, inputs: str | dict[str, Any], use_knowledge: bool=True, task_type: str="general", use_xml_tags_in_prompt=True):
    # temporary remove chat template
    if role == "proposer":
        # if use_knowledge:
        if task_type == "general":
            sys_prompt = PROPOSER_PROMPT_WITH_KNOWLEDGE
        elif task_type == "creative_writing":
            sys_prompt = PROPOSER_PROMPT_WITH_KNOWLEDGE_CREATIVE_WRITING
        elif task_type == "health_care":
            sys_prompt = PROPOSER_PROMPT_WITH_KNOWLEDGE_HEALTH_CARE
        elif task_type == "knowledge_elicitation":
            # raise Exception("Prompt is outdated")
            sys_prompt = PROPOSER_PROMPT_WITH_KNOWLEDGE_ELICITATION_VERIFIABLE
        elif task_type == "math":
            sys_prompt = PROPOSER_PROMPT_WITH_KNOWLEDGE_MATH
        elif task_type == "reasoning":
            raise Exception("Prompt is outdated")
            # sys_prompt = PROPOSER_PROMPT_WITH_KNOWLEDGE_REASONING
        else:
            raise Exception("task_type {} is invalid".format(task_type))

        if task_type == "creative_writing":
            user_prompt = PROPOSER_USER_PROMPT_WITH_KNOWLEDGE_CREATIVE_WRITING.replace("{knowledge}", inputs)
        else:
            if use_xml_tags_in_prompt:

                user_prompt = PROPOSER_USER_PROMPT_WITH_KNOWLEDGE.replace("{knowledge}", inputs)
            else:
                user_prompt = PROPOSER_USER_PROMPT_WITH_KNOWLEDGE_NO_XML.replace("{knowledge}", inputs)
        # else:
        #     sys_prompt = PROPOSER_PROMPT_WITHOUT_KNOWLEDGE
        #     user_prompt = PROPOSER_USER_PROMPT_WITHOUT_KNOWLEDGE
    elif role == "solver":
        if task_type == "general":
            sys_prompt = SOLVER_SYSTEM_PROMPT
        elif task_type == "creative_writing":
            sys_prompt = SOLVER_SYSTEM_PROMPT_CREATIVE_WRITING
        elif task_type == "health_care":
            sys_prompt = SOLVER_SYSTEM_PROMPT_HEALTH_CARE
        elif task_type == "knowledge_elicitation":
            sys_prompt = SOLVER_SYSTEM_PROMPT_ELICITATION_VERIFIABLE
        elif task_type == "math":
            sys_prompt = SOLVER_SYSTEM_PROMPT_MATH
        else:
            raise Exception("task_type {} is invalid".format(task_type))
        user_prompt = SOLVER_USER_PROMPT.replace("{question}", inputs)
    elif role == "verifier":
        sys_prompt = VERIFIER_SYSTEM_PROMPT if use_xml_tags_in_prompt else VERIFIER_SYSTEM_PROMPT_NO_XML
        if use_xml_tags_in_prompt:
            answers = "\n\n".join(["<candidate_answer_{}>\n{}\n</candidate_answer_{}>".format(r_idx+1, r, r_idx+1) for r_idx, r in enumerate(inputs["answers"])])
        else:
            answers = "\n\n".join(["Answer {}:\n{}".format(r_idx+1, r) for r_idx, r in enumerate(inputs["answers"])]) + "\n\n"
        
        user_prompt = (VERIFIER_USER_PROMPT if use_xml_tags_in_prompt else VERIFIER_USER_PROMPT_NO_XML).replace("{knowledge}", inputs["knowledge"]).replace("{question}", inputs["question"]).replace("{ref_answer}", inputs["ref_answer"]).replace("{answers}", answers)
    elif role == "verifier_scoring":
        sys_prompt = VERIFIER_SCORING_SYSTEM_PROMPT if use_xml_tags_in_prompt else VERIFIER_SCORING_SYSTEM_PROMPT_NO_XML
        user_prompt = (VERIFIER_SCORING_USER_PROMPT if use_xml_tags_in_prompt else VERIFIER_SCORING_USER_PROMPT_NO_XML).replace("{question}", inputs["question"]).replace("{answer}", inputs["answer"]).replace("{rubric}", inputs["rubric"])
    elif role == "verifier_scoring_pairwise":
        sys_prompt = VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE if use_xml_tags_in_prompt else VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE_NO_XML
        user_prompt = (VERIFIER_SCORING_USER_PROMPT_PAIRWISE if use_xml_tags_in_prompt else VERIFIER_SCORING_USER_PROMPT_PAIRWISE_NO_XML).replace("{question}", inputs["question"]).replace("{answer_a}", inputs["answer_a"]).replace("{answer_b}", inputs["answer_b"]).replace("{rubric}", inputs["rubric"])
    elif role == "verifier_scoring_pairwise_anchor":
        sys_prompt = VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE_ANCHOR if use_xml_tags_in_prompt else VERIFIER_SCORING_SYSTEM_PROMPT_PAIRWISE_ANCHOR_NO_XML
        user_prompt = (VERIFIER_SCORING_USER_PROMPT_PAIRWISE_ANCHOR if use_xml_tags_in_prompt else VERIFIER_SCORING_USER_PROMPT_PAIRWISE_ANCHOR_NO_XML).replace("{question}", inputs["question"]).replace("{answer_a}", inputs["answer_a"]).replace("{answer_b}", inputs["answer_b"]).replace("{rubric}", inputs["rubric"])

    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    new_prompt = None

    if tokenizer is not None:
        new_prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        ) # Qwen3 tokenizer has an argument "enable_thinking", which is turned on by default

        # TODO: Truncate msgs too
        new_prompt_tokens = tokenizer.encode(new_prompt, add_special_tokens=False)
        if len(new_prompt_tokens) > 16384:
            print("Prompt has more than 16384 tokens. Truncating from the left.")
            new_prompt = tokenizer.decode(new_prompt_tokens[-16384:])


    return new_prompt, msgs

def get_solver_response(llm, tokenizer, inputs, solver_sampling_params, use_peft, peft_dir, task_type, teacher_model, client=None, use_xml_tags_in_prompt=True):
    solver_prompt, solver_msgs = get_prompt(role="solver", tokenizer=tokenizer, inputs=inputs, task_type=task_type, use_xml_tags_in_prompt=use_xml_tags_in_prompt)
    
    assert (llm is None) != (teacher_model is None)
    cost = 0
    if teacher_model is None:
    
        solver_output_texts = generate(llm, solver_msgs, solver_sampling_params, use_peft, peft_dir, client=client)
    else:
        print("==========Generating solver responses with teacher model {}==========\n\n".format(teacher_model))
        solver_output_texts, cost = openai_generate(model=teacher_model, messages=solver_msgs, sampling_params=solver_sampling_params, client=client)

    res = []


    for solver_output_text in solver_output_texts:
        answer = extract_answer_solver(solver_output_text)


        solver_data = {
            "solver_answer": "",
            "solver_conversations": [
                {
                    "role": "",
                    "content": "",
                }
            ],
            "is_solver_valid": 0,
            "verifier_scoring_evaluation": [
                {
                    "name": "",
                    "thoughts": "",
                    "rating": -1.0,
                }
            ],
            "verifier_scoring_score": -1.0,
            "verifier_scoring_conversations": [
                {
                    "role": "",
                    "content": "",
                }
            ],
            "is_verifier_scoring_valid": 0,
        }
        if not answer:
            continue
        solver_data.update({
            "solver_answer": answer,
            "solver_conversations": solver_msgs + [{"role": "assistant", "content": solver_output_text}],
            "is_solver_valid": 1
        })

        res.append(solver_data)
            
    return res, cost


def get_verifier_scoring_response(llm, tokenizer, inputs, verifier_scoring_sampling_params, use_peft, peft_dir, task_type, teacher_model, client=None, use_xml_tags_in_prompt=True):
    verifier_scoring_prompt, verifier_scoring_msgs = get_prompt(role="verifier_scoring", tokenizer=tokenizer, inputs=inputs, task_type=task_type, use_xml_tags_in_prompt=use_xml_tags_in_prompt)


    assert (llm is None) != (teacher_model is None)
    cost = 0
    if teacher_model is None:
        verifier_scoring_output_texts = generate(llm, verifier_scoring_msgs, verifier_scoring_sampling_params, use_peft, peft_dir, client=client)
        verifier_scoring_output_text = verifier_scoring_output_texts[0]
    else:
        print("==========Generating verifier scoring with teacher model {}==========\n\n".format(teacher_model))
        verifier_scoring_output_texts, cost = openai_generate(model=teacher_model, messages=verifier_scoring_msgs, sampling_params=verifier_scoring_sampling_params, client=client)
        verifier_scoring_output_text = verifier_scoring_output_texts[0]

    score, evaluation = extract_rubric_score_from_evaluation(verifier_scoring_output_text, inputs["rubric"])
    
    res = {
        "is_verifier_scoring_valid": 0,
    }

    if score == -1.0 or not evaluation:
        return res, cost

    res = {
        "verifier_scoring_evaluation": [evaluation[c] for c in evaluation],
        "verifier_scoring_score": float(score),
        "verifier_scoring_conversations": verifier_scoring_msgs + [{"role": "assistant", "content": verifier_scoring_output_text}],
        "is_verifier_scoring_valid": 1,
    }
            
    return res, cost

def get_verifier_scoring_response_pairwise(llm, tokenizer, inputs, verifier_scoring_sampling_params, use_peft, peft_dir, task_type, teacher_model, client=None, use_xml_tags_in_prompt=True):
    i_better = []

    cost = 0
    for turn in ["ij", "ji"]:
        turn_inputs = {
            "question": inputs["question"],
            "rubric": inputs["rubric"],
        }
        if turn == "ij":
            turn_inputs["answer_a"] = inputs["answer_i"]
            turn_inputs["answer_b"] = inputs["answer_j"]
        else:
            turn_inputs["answer_a"] = inputs["answer_j"]
            turn_inputs["answer_b"] = inputs["answer_i"]


        verifier_scoring_prompt, verifier_scoring_msgs = get_prompt(role="verifier_scoring_pairwise", tokenizer=tokenizer, inputs=turn_inputs, task_type=task_type, use_xml_tags_in_prompt=use_xml_tags_in_prompt)


        assert (llm is None) != (teacher_model is None)
        if teacher_model is None:
            verifier_scoring_output_texts = generate(llm, verifier_scoring_msgs, verifier_scoring_sampling_params, use_peft, peft_dir, client=client)
            verifier_scoring_output_text = verifier_scoring_output_texts[0]
        else:
            verifier_scoring_output_texts, turn_cost = openai_generate(model=teacher_model, messages=verifier_scoring_msgs, sampling_params=verifier_scoring_sampling_params, client=client)
            verifier_scoring_output_text = verifier_scoring_output_texts[0]
            cost = cost + turn_cost

        answer_a_better = extract_rubric_score_pairwise(verifier_scoring_output_text)
        if answer_a_better < 0:
            # If 1 of the turn is invalid, the entire comparison is treated to be invalid
            return [-1.0, -1.0]
        if turn == "ij":
            i_better.append(answer_a_better)
        else:
            i_better.append(1.0 - answer_a_better)
    
    return i_better, cost

def get_verifier_scoring_response_pairwise_anchor(llm, tokenizer, inputs, verifier_scoring_sampling_params, use_peft, peft_dir, task_type, teacher_model, client=None, use_xml_tags_in_prompt=True):
    scores = []
    conversations = []
    
    cost = 0
    for turn in ["ij", "ji"]:
        turn_inputs = {
            "question": inputs["question"],
            "rubric": inputs["rubric"],
        }
        if turn == "ij":
            turn_inputs["answer_a"] = inputs["answer_i"]
            turn_inputs["answer_b"] = inputs["answer_j"]
        else:
            turn_inputs["answer_a"] = inputs["answer_j"]
            turn_inputs["answer_b"] = inputs["answer_i"]


        verifier_scoring_prompt, verifier_scoring_msgs = get_prompt(role="verifier_scoring_pairwise_anchor", tokenizer=tokenizer, inputs=turn_inputs, task_type=task_type, use_xml_tags_in_prompt=use_xml_tags_in_prompt)


        assert (llm is None) != (teacher_model is None)
        if teacher_model is None:
            verifier_scoring_output_texts = generate(llm, verifier_scoring_msgs, verifier_scoring_sampling_params, use_peft, peft_dir, client=client)
            verifier_scoring_output_text = verifier_scoring_output_texts[0]
        else:
            verifier_scoring_output_texts, turn_cost = openai_generate(model=teacher_model, messages=verifier_scoring_msgs, sampling_params=verifier_scoring_sampling_params, client=client)
            verifier_scoring_output_text = verifier_scoring_output_texts[0]
            cost = cost + turn_cost

        turn_scores = extract_rubric_score_pairwise_anchor(verifier_scoring_output_text)
        if turn_scores is None:
            # If 1 of the turn is invalid, the entire comparison is treated to be invalid
            return {"is_verifier_scoring_valid": 0}, cost

        conversations.extend(verifier_scoring_msgs + [{"role": "assistant", "content": verifier_scoring_output_text}])

        if turn == "ij":
            scores.append(turn_scores["a"] - turn_scores["b"])
        else:
            scores.append(turn_scores["b"] - turn_scores["a"])
    
    res = {
        "verifier_scoring_score": sum(scores)/len(scores),
        "verifier_scoring_conversations": conversations,
        "is_verifier_scoring_valid": 1,
    }

    return res, cost


def check_args(args):
    assert args.generator_name is not None

    if args.existing_questions_path is not None:
        assert args.use_teacher_model not in ["proposer"], "Cannot regenerate problems using teacher model if reusing problems"

    if args.existing_questions_path is None:
        assert args.train_set_size is not None

    if args.use_existing_solver_responses:
        # Use case: use gpt to regenerate rubric and/or scores
        # If we reuse the solver responses, we must be reusing the questions too.
        assert args.existing_questions_path is not None, "Problems need to be fixed to reuse solver responses"
        assert args.use_teacher_model not in ["proposer"], "Problems need to be fixed to reuse solver responses"
        assert args.use_teacher_model not in ["solver"], "Cannot regenerate solver responses using teacher model if reusing solver responses"
    if args.use_existing_rubrics:
        # Use case: use gpt to regenerate scores
        assert args.existing_questions_path is not None, "Problems need to be fixed to reuse rubrics"
        assert args.use_teacher_model not in ["proposer", "solver"], "Cannot regenerate problems or responses using the teacher model. Problems and responses need to be fixed to reuse rubrics"
        assert args.use_teacher_model not in ["rubric", "rubric_and_scoring"], "Cannot regenerate rubrics using teacher model if reusing rubrics"
    # Doesnt make sense to reuse the scores since in that case the questions, responses, and rubrics have to be reused as well and nothing new is generated

    if "writing" in args.task_type:
        assert args.no_xml_tags_in_prompt is False, "Tasks related to writing has to use xml tags in the prompt"

def get_letter(use_existing, existing_questions_path, use_teacher_model, role, task_type):
    letter = ""
    if existing_questions_path is not None and use_existing:
        idx = 0
        if role == "proposer":
            idx = 0
        elif role == "solver":
            idx = 1
        elif role == "rubric":
            idx = 2
        elif role == "scoring":
            idx = 3
        else:
            raise Exception("role {} is not valid".format(role))
        letter = existing_questions_path.split("{}_".format(task_type))[-1][idx]
    elif use_teacher_model is not None and role in use_teacher_model:
        letter = "t"
    else:
        letter = "s"
    return letter


def get_questions_per_k_count(res):
    if len(res) == 0:
        return 0
    
    curr_idx = len(res) - 2
    last_k = res[-1]["knowledge"]
    while curr_idx >= 0 and res[curr_idx]["knowledge"] == last_k:
        curr_idx = curr_idx - 1
    return len(res) - (curr_idx + 1)



def is_server_running(port=8000):
    url = f"http://localhost:{port}/v1/models"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Server is running.")
            return True
    except requests.ConnectionError:
        pass
    return False

def setup_model(args):
    proposer_num_sampling_sequences = 1
    no_vllm = (args.existing_questions_path is not None or args.teacher_model in ["proposer"]) and (args.use_existing_solver_responses or args.teacher_model in ["solver"]) and (args.use_existing_rubrics or args.use_teacher_model in ["rubric", "rubric_and_scoring"]) and (args.use_teacher_model in ["scoring", "rubric_and_scoring"])

    llm = None
    tokenizer = None
    if not no_vllm:
        for _ in tqdm(
            range(300), desc="Waiting for the server to start."
        ):  # Retry for up to ~300 seconds
            if is_server_running(port=args.vllm_port):
                break
            time.sleep(5)
        else:
            print("Server did not start in time.")
            exit(1)

        llm = args.generator_name

    proposer_sampling_params = SamplingParams(
        n=proposer_num_sampling_sequences,
        max_tokens=6144,
        temperature=args.proposer_solver_temperature,
        top_p=1.0,
        # seed=sampling_seed,
    )

    solver_sampling_params = SamplingParams(
        n=args.solver_num_sampling_sequences,
        max_tokens=6144,
        temperature=args.proposer_solver_temperature,
        top_p=1.0,
        # seed=sampling_seed,
    )

    verifier_sampling_params = SamplingParams(
        n=1,
        max_tokens=8192,
        temperature=0.0,
        top_p=1.0,
        # seed=sampling_seed,
    )

    verifier_scoring_sampling_params = SamplingParams(
        n=1,
        max_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        # seed=sampling_seed,
    )
    return llm, tokenizer, proposer_sampling_params, solver_sampling_params, verifier_sampling_params, verifier_scoring_sampling_params


def get_res_path(args):

    use_peft = args.peft_dir is not None and len(args.peft_dir) > 0

    generator = args.generator_name

    temp_suffix = "_temp_{:.0e}".format(args.proposer_solver_temperature)
    proposer_num_sampling_sequences = 1
    num_seq_suffix = "_num_seq_{}_{}".format(proposer_num_sampling_sequences, args.solver_num_sampling_sequences)
    

    pl = get_letter(use_existing=args.existing_questions_path is not None, existing_questions_path=args.existing_questions_path, use_teacher_model=args.use_teacher_model, role="proposer", task_type=args.task_type)
    sl = get_letter(use_existing=args.use_existing_solver_responses, existing_questions_path=args.existing_questions_path, use_teacher_model=args.use_teacher_model, role="solver", task_type=args.task_type)
    vl = get_letter(use_existing=args.use_existing_rubrics, existing_questions_path=args.existing_questions_path, use_teacher_model=args.use_teacher_model, role="rubric", task_type=args.task_type)
    srl = get_letter(use_existing=False, existing_questions_path=args.existing_questions_path, use_teacher_model=args.use_teacher_model, role="scoring", task_type=args.task_type)

    teacher_model_suffix = pl + sl + vl + srl

    if "t" in teacher_model_suffix:
        teacher_model_suffix = "{}_{}".format(teacher_model_suffix, args.teacher_model)
    
    scoring_mode_suffix = ""
    if args.scoring_mode != "pointwise":
        scoring_mode_suffix = "_{}".format(args.scoring_mode)
    
    res_path = os.path.join(args.output_dir, "{}{}_{}_{}{}.jsonl".format(generator, args.other_suffix, args.task_type, teacher_model_suffix, scoring_mode_suffix))

    print("Result path is {}".format(res_path))
    return res_path


def get_cached_results(res_path):
    res = []

    if os.path.exists(res_path):
        with open(res_path, "r") as src_json:
            if res_path.endswith(".jsonl"):
                res = [json.loads(line) for line in src_json]
            else:
                res = json.load(src_json)
    return res

def get_existing_questions(existing_questions_path):
    existing_questions = None
    if existing_questions_path:
        with open(existing_questions_path, "r") as ext_json:
            if existing_questions_path.endswith(".jsonl"):
                existing_questions = [json.loads(line) for line in ext_json]
            else:
                existing_questions = json.load(ext_json)

            existing_questions = [eq for eq in existing_questions if eq["is_proposer_valid"] == 1]
    return existing_questions


def prefill_knowledge(res, existing_questions, args):
    start_idx = len(res)
    print("==========Fast forwarding to the {}th example==========".format(start_idx))


    total_size = args.train_set_size if existing_questions is None else len(existing_questions)

    for i in range(total_size):

        if args.existing_questions_path is not None:
            eq = existing_questions[i]

        if args.existing_questions_path is None and args.use_knowledge:

            questions_per_k_count = get_questions_per_k_count(res)
            if questions_per_k_count % args.num_questions_per_k == 0:
                print("==========Sampling new knowledge==========")
                knowledge = knowledge_base.sample(1)[0]
            else:
                knowledge = res[-1]["knowledge"]
        elif args.existing_questions_path is not None:
            knowledge = eq["knowledge"]
        elif not args.use_knowledge:
            knowledge = "None"
        
        if i < start_idx:
            # Fast forward to the correct random seed
            continue
        
        gt_field_name = "ground_truth" if args.no_xml_tags_in_prompt else "gold"

        new_example = {
            "knowledge": "",
            "proposer_problem": "",
            "proposer_ref_answer": "",
            "proposer_conversations": [
                {
                    "role": "",
                    "content": "",
                }
            ],
            "is_proposer_valid": 0,

            "solver_responses": [
                {
                    "solver_answer": "",
                    "solver_conversations": [
                        {
                            "role": "",
                            "content": "",
                        }
                    ],
                    "is_solver_valid": 0,
                    "verifier_scoring_evaluation": [
                        {
                            "name": "",
                            "thoughts": "",
                            "rating": -1.0,
                        }
                    ],
                    "verifier_scoring_score": -1.0,
                    "verifier_scoring_conversations": [
                        {
                            "role": "",
                            "content": "",
                        }
                    ],
                    "is_verifier_scoring_valid": 0,
                }
            ],

            "verifier_rubric": [
                {
                    "name": "",
                    gt_field_name: "", # originally "ground_truth"
                    "description": "",
                    "weight": -1.0,
                }
            ],
            "verifier_rubric_text": "",
            "verifier_conversations": [
                {
                    "role": "",
                    "content": "",
                }
            ],
            "is_verifier_valid": 0,
            "cost": 0.0
        }

        new_example["knowledge"] = knowledge


        if args.existing_questions_path is not None:
            for key in ["proposer_problem", "proposer_ref_answer", "proposer_conversations", "is_proposer_valid"]:
                new_example[key] = eq[key]
        
        if args.use_existing_solver_responses:
            valid_solver_res = [
                {
                    "solver_answer": sr["solver_answer"],
                    "solver_conversations": sr["solver_conversations"],
                    "is_solver_valid": sr["is_solver_valid"],
                    "verifier_scoring_evaluation": [
                        {
                            "name": "",
                            "thoughts": "",
                            "rating": -1.0,
                        }
                    ],
                    "verifier_scoring_score": -1.0,
                    "verifier_scoring_conversations": [
                        {
                            "role": "",
                            "content": "",
                        }
                    ],
                    "is_verifier_scoring_valid": 0,
                } for sr in eq["solver_responses"]
            ]
            new_example["solver_responses"] = valid_solver_res
        
        if args.use_existing_rubrics:
            for key in ["verifier_rubric", "verifier_rubric_text", "verifier_conversations", "is_verifier_valid"]:
                new_example[key] = eq[key]

        res.append(new_example)
    return res, start_idx

def run_pipeline(example, idx, **kwargs):
    # Retrying logic:
    # Proposer and Solver are retried for 5 times, Verifier and Verifier Scoring are not retried since they use greedy decoding.

    args = kwargs["run_args"]

    retry_limit = kwargs["retry_limit"]

    use_peft = args.peft_dir is not None and len(args.peft_dir) > 0

    llm = kwargs["llm"]
    tokenizer = kwargs["tokenizer"]

    proposer_sampling_params = kwargs["proposer_sampling_params"]
    solver_sampling_params = kwargs["solver_sampling_params"]
    verifier_sampling_params = kwargs["verifier_sampling_params"]
    verifier_scoring_sampling_params = kwargs["verifier_scoring_sampling_params"]

    num_solver_responses_per_q = kwargs["num_solver_responses_per_q"]

    print("==========Proposer generation starts==========")
    knowledge = example["knowledge"]

    global vllm_client
    global openai_client

    if args.existing_questions_path is None:        

        proposer_prompt, proposer_msgs = get_prompt(role="proposer", tokenizer=tokenizer, inputs=knowledge, task_type=args.task_type, use_xml_tags_in_prompt=not args.no_xml_tags_in_prompt)

        retry_count = 0
        is_proposer_valid = 0
        
        while not is_proposer_valid and retry_count < retry_limit:
            print("Proposer retry count {}".format(retry_count))

            if args.use_teacher_model not in ["proposer"]:
                # import pdb; pdb.set_trace()
                proposer_output_texts = generate(llm, proposer_msgs, proposer_sampling_params, use_peft, args.peft_dir, vllm_port=args.vllm_port, client=vllm_client)

            else:
                print("==========Generating problem with teacher model {}==========\n\n".format(args.teacher_model))
                proposer_output_texts, cost = openai_generate(model=args.teacher_model, messages=proposer_msgs, sampling_params=proposer_sampling_params, client=openai_client)
                example["cost"] = example["cost"] + cost

            retry_count = retry_count + 1

            assert len(proposer_output_texts) == 1

            proposer_output_text = proposer_output_texts[0]

            # Find the first that is valid.

            question = extract_question(proposer_output_text)
            ref_answer = extract_answer_proposer(proposer_output_text)

            proposer_conversations = proposer_msgs + [{"role": "assistant", "content": proposer_output_text}]

            if not question:
                print("No valid question from proposer. Retrying", flush=True)
                continue

            if not ref_answer:
                print("No valid reference answer from proposer. Retrying", flush=True)
                continue

            is_proposer_valid = 1
            break

    else:
        print("==========Taking proposer generation from existing question {}==========\n\n".format(idx))
        is_proposer_valid = example["is_proposer_valid"]

    if not is_proposer_valid:
        print("No valid question or ref answer from proposer {}, but will keep the new example as is to avoid repeatedly calling the teacher model and incurring excessive cost.".format(proposer_conversations[-1]["content"]), flush=True)
        return example

    if args.existing_questions_path is None:
        example.update(
            {
                "knowledge": knowledge if knowledge is not None else "",
                "proposer_problem": question,
                "proposer_ref_answer": ref_answer,
                "proposer_conversations": proposer_conversations,
                "is_proposer_valid": is_proposer_valid
            }
        )

    print("==========Proposer generation completed with {} valid response==========\n\n".format(int(is_proposer_valid)))




    print("==========Solver generation starts==========")
    if not args.use_existing_solver_responses:
        retry_count = 0
        
        valid_solver_res = []
        
        generate_solver_response_with_teacher = args.use_teacher_model in ["solver"]

        while len(valid_solver_res) <= num_solver_responses_per_q and retry_count < retry_limit:
            print("Solver retry count {}".format(retry_count))
            solver_res, cost = get_solver_response(
                None if generate_solver_response_with_teacher else llm, 
                tokenizer, 
                example["proposer_problem"], 
                solver_sampling_params, 
                use_peft, 
                args.peft_dir, 
                task_type=args.task_type, 
                teacher_model=args.teacher_model if generate_solver_response_with_teacher else None,
                client=openai_client if generate_solver_response_with_teacher else vllm_client,
                use_xml_tags_in_prompt=not args.no_xml_tags_in_prompt,
            )

            example["cost"] = example["cost"] + cost

            valid_solver_res.extend([s for s in solver_res if s["is_solver_valid"]])
            retry_count = retry_count + 1



    else:
        print("==========Taking solver responses from existing question {}==========\n\n".format(idx))
        valid_solver_res = [sr for sr in example["solver_responses"] if sr["is_solver_valid"]]

    if len(valid_solver_res) == 0:
        print("No valid response from solver, but will keep the new example as is to avoid repeatedly calling the teacher model and incurring excessive cost.", flush=True)
        return example
            
    if not args.use_existing_solver_responses:
        example.update(
            {
                "solver_responses": valid_solver_res,
            }
        )

    print("==========Solver generation completed with {} valid responses==========\n\n".format(len(valid_solver_res)))



    print("==========Verifier generation starts==========")
    if not args.use_existing_rubrics:
        solver_answers_for_rubrics = [sr["solver_answer"] for sr in example["solver_responses"] if sr["is_solver_valid"]]
        solver_answers_for_rubrics.sort(key=lambda ans: len(ans))
        total_num_valid_answers = len(solver_answers_for_rubrics)

        # Response for writing task is longer, so need to truncate the responses and reduce the number of responses to feed into the verifier
        if "writing" in args.task_type:
            slice_step = max(total_num_valid_answers//4, 1)
            solver_answers_for_rubrics_truncated = []
            # print("===============\n[verifier generation]: solver_answers_for_rubrics[::slice_step]:\n\n{}\n===============".format(solver_answers_for_rubrics[::slice_step]))
            for solver_answer in solver_answers_for_rubrics[::slice_step]:
                solver_answers_for_rubrics_truncated.append(" ".join(solver_answer.split(" ")[:2048]))
            solver_answers_for_rubrics = solver_answers_for_rubrics_truncated
            
        else:
            slice_step = max(total_num_valid_answers//10, 1)
            solver_answers_for_rubrics = solver_answers_for_rubrics[::slice_step]

        # random.shuffle(solver_answers_for_rubrics)

        include_know_and_ref_ans_in_rubric = not args.no_know_in_rubric and args.use_knowledge

        verifier_prompt, verifier_msgs = get_prompt(
            role="verifier", 
            tokenizer=tokenizer, 
            inputs={
                "knowledge": example["knowledge"] if include_know_and_ref_ans_in_rubric else "None", 
                "question": example["proposer_problem"], 
                "ref_answer": example["proposer_ref_answer"] if include_know_and_ref_ans_in_rubric else "None",
                "answers": solver_answers_for_rubrics # We only take a part of the responses. Otherwise the context will explode.
            },
            task_type=args.task_type,
            use_xml_tags_in_prompt=not args.no_xml_tags_in_prompt
        )
        if args.use_teacher_model not in ["rubric", "rubric_and_scoring"]:
            verifier_completions = generate(llm, verifier_msgs, verifier_sampling_params, use_peft, args.peft_dir, vllm_port=args.vllm_port, client=vllm_client)
            
        else:
            print("==========Generating rubrics with teacher model {}==========\n\n".format(args.teacher_model))
            verifier_completions, cost = openai_generate(model=args.teacher_model, messages=verifier_msgs, sampling_params=verifier_sampling_params, client=openai_client)
            example["cost"] = example["cost"] + cost

        verifier_completion = verifier_completions[0]

        rubric_text, rubric = extract_rubric(verifier_completion)


        if not rubric_text or not rubric:
            is_verifier_valid = 0
        else:
            is_verifier_valid = 1
        
        verifier_conversations = verifier_msgs + [{"role": "assistant", "content": verifier_completion}]
    else:
        print("==========Taking rubrics from existing question {}==========\n\n".format(idx))
        is_verifier_valid = example["is_verifier_valid"]

    if not is_verifier_valid:
        print("No valid rubric from verifier, but will keep the new example as is to avoid repeatedly calling the teacher model and incurring excessive cost.", flush=True)
        return example

    if not args.use_existing_rubrics:
        example.update(
            {
                "verifier_rubric": [rubric[c] for c in rubric],
                "verifier_rubric_text": rubric_text,
                "verifier_conversations": verifier_conversations,
                "is_verifier_valid": is_verifier_valid
            }
        )

    print("==========Verifier generation completed with {} valid response==========\n\n".format(int(is_verifier_valid)))









    print("==========Verifier scoring generation starts==========")

    generate_verifier_scoring_with_teacher = args.use_teacher_model in ["scoring", "rubric_and_scoring"]
    if args.scoring_mode == "pairwise":
        comparison_matrix = [[-1.0] * len(example["solver_responses"]) for _ in len(example["solver_responses"])]


        for i_vs, solver_data_i in enumerate(example["solver_responses"]):
            i_vs_comparisons = []
            for j_vs, solver_data_j in enumerate(example["solver_responses"][i_vs+1:]):
                if solver_data_i["solver_answer"].strip() == solver_data_j["solver_answer"].strip():
                    i_vs_comparisons.append([0.5, 0.5])
                    comparison_matrix[i_vs][j_vs] = 0.5
                    comparison_matrix[j_vs][i_vs] = 0.5
                    continue
                verifier_scoring_res, cost = get_verifier_scoring_response_pairwise(
                    None if generate_verifier_scoring_with_teacher else llm, 
                    tokenizer,
                    inputs={
                        "question": example["proposer_problem"],
                        "answer_i": solver_data_i["solver_answer"],
                        "answer_j": solver_data_j["solver_answer"],
                        "rubric": example["verifier_rubric_text"],
                    },
                    verifier_scoring_sampling_params=verifier_scoring_sampling_params,
                    use_peft=use_peft,
                    peft_dir=args.peft_dir,
                    task_type=args.task_type,
                    teacher_model=args.teacher_model if generate_verifier_scoring_with_teacher else None,
                    client=openai_client if generate_verifier_scoring_with_teacher else vllm_client,
                    use_xml_tags_in_prompt=not args.no_xml_tags_in_prompt
                )

                example["cost"] = example["cost"] + cost

                if -1.0 in verifier_scoring_res["i_better"]:
                    print("No valid comparison for solver response {} and {}. Skipping".format(i_vs, j_vs))
                    i_vs_comparisons.append(verifier_scoring_res["i_better"])
                    continue

                i_vs_comparisons.append(verifier_scoring_res["i_better"])
                i_better_avg = sum(verifier_scoring_res["i_better"])/len(verifier_scoring_res["i_better"])
                comparison_matrix[i_vs][j_vs] = i_better_avg
                comparison_matrix[j_vs][i_vs] = 1 - i_better_avg


            valid_comparisons = [comparison for comparison in comparison_matrix[i_vs] if comparison >= 0.0]
            is_verifier_scoring_valid = len(valid_comparisons) > 0

            if not is_verifier_scoring_valid:
                print("No valid scores for solver response {}. Skipping".format(i_vs))
                continue

            example["solver_responses"][i_vs]["is_verifier_scoring_valid"] = 1

            example["solver_responses"][i_vs]["verifier_scoring_conversations"] = [{"role": "assistant", "content": str(i_vs_comparisons)}]

            example["solver_responses"][i_vs]["verifier_scoring_score"] = sum(valid_comparisons)/len(valid_comparisons)

    elif args.scoring_mode == "pairwise_anchor":
        solver_answers = [solver_data["solver_answer"] for solver_data in example["solver_responses"] if solver_data["is_solver_valid"]]
        solver_answers.sort(key=lambda x: len(x))
        anchor_answer = solver_answers[len(solver_answers)//2]

        for i_vs, solver_data in enumerate(example["solver_responses"]):
            
            verifier_scoring_res, cost = get_verifier_scoring_response_pairwise_anchor(
                None if generate_verifier_scoring_with_teacher else llm, 
                tokenizer,
                inputs={
                    "question": example["proposer_problem"],
                    "answer_i": solver_data["solver_answer"],
                    "answer_j": anchor_answer,
                    "rubric": example["verifier_rubric_text"],
                },
                verifier_scoring_sampling_params=verifier_scoring_sampling_params,
                use_peft=use_peft,
                peft_dir=args.peft_dir,
                task_type=args.task_type,
                teacher_model=args.teacher_model if generate_verifier_scoring_with_teacher else None,
                client=openai_client if generate_verifier_scoring_with_teacher else vllm_client,
                use_xml_tags_in_prompt=not args.no_xml_tags_in_prompt
            )

            example["cost"] = example["cost"] + cost

            if verifier_scoring_res["is_verifier_scoring_valid"] == 0:
                print("No valid scores for solver response {}. Skipping".format(i_vs))
                continue
            
            if verifier_scoring_res["is_verifier_scoring_valid"] == 1:
                example["solver_responses"][i_vs].update(
                    verifier_scoring_res
                )
            
    elif args.scoring_mode == "pointwise":

        for i_vs, solver_data in enumerate(example["solver_responses"]):
            curr_vs_valid_count = 0
            # retry_count = 0

            # No use to retry since verifier uses greedy decoding.
            # is_verifier_scoring_valid = False
            # while not is_verifier_scoring_valid and retry_count < retry_limit:
            #     print("Verifier Scoring retry count {} for solver response {}".format(retry_count, i_vs))

            verifier_scoring_res, cost = get_verifier_scoring_response(
                None if generate_verifier_scoring_with_teacher else llm , 
                tokenizer,
                inputs={
                    "question": example["proposer_problem"],
                    "answer": solver_data["solver_answer"],
                    "rubric": example["verifier_rubric_text"],
                },
                verifier_scoring_sampling_params=verifier_scoring_sampling_params,
                use_peft=use_peft,
                peft_dir=args.peft_dir,
                task_type=args.task_type,
                teacher_model=args.teacher_model if generate_verifier_scoring_with_teacher else None,
                client=openai_client if generate_verifier_scoring_with_teacher else vllm_client,
                use_xml_tags_in_prompt=not args.no_xml_tags_in_prompt
            )

            example["cost"] = example["cost"] + cost


            # retry_count = retry_count + 1

            if verifier_scoring_res["is_verifier_scoring_valid"] == 0:
                print("No valid scores for solver response {}. Skipping".format(i_vs))
                continue
            
            if verifier_scoring_res["is_verifier_scoring_valid"] == 1:
                example["solver_responses"][i_vs].update(
                    verifier_scoring_res
                )
    
    num_valid_scoring = sum([solver_data["is_verifier_scoring_valid"] for solver_data in example["solver_responses"]])

    if num_valid_scoring < num_solver_responses_per_q:
        # if args.use_teacher_model is not None:
        print("Not enough valid scores for solver responses. Expect {}; Got {}, but will keep the new example as is to avoid repeatedly calling the teacher model and incurring excessive cost.".format(num_solver_responses_per_q, num_valid_scoring), flush=True)
        return example

    
    print("==========Verifier scoring generation completed with {} valid responses==========\n\n".format(num_valid_scoring))

    return example
    

    # step_count = step_count + 1
    
    # if step_count % save_interval == 0 and step_count > 0:
    #     with open(res_path, "w") as res_json:
    #         json.dump(res, res_json, indent=4)

def get_sizes(completed_dataset):
    valid_proposer_count = sum([example["is_proposer_valid"] for example in completed_dataset])
    valid_solver_count = sum([s["is_solver_valid"] for example in completed_dataset for s in example["solver_responses"]])
    valid_verifier_count = sum([example["is_verifier_valid"] for example in completed_dataset])
    valid_verifier_scoring_count = sum([s["is_verifier_scoring_valid"] for example in completed_dataset for s in example["solver_responses"]])

    return valid_proposer_count, valid_solver_count, valid_verifier_count, valid_verifier_scoring_count

def get_total_costs(completed_dataset):
    return sum([example["cost"] for example in completed_dataset])

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--peft_dir", type=str)
    parser.add_argument("--other_suffix", type=str, default="")
    parser.add_argument("--use_knowledge", type=int, default=1, help="Use knowledge or not")
    parser.add_argument("--train_set_size", type=int, default=None)
    parser.add_argument("--task_type", type=str, choices=["general", "knowledge_elicitation", "reasoning", "health_care", "creative_writing", "math"], default="general",
                        help="Type of the task")
    parser.add_argument("--existing_questions_path", type=str, default=None)
    parser.add_argument("--use_existing_solver_responses", action="store_true")
    parser.add_argument("--use_existing_rubrics", action="store_true")
    parser.add_argument("--generator_name", type=str, default="")
    parser.add_argument("--knowledge_base", type=str, default="Salesforce/wikitext")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--teacher_model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--use_teacher_model", type=str, choices=["proposer", "solver", "rubric", "scoring", "rubric_and_scoring"], default=None)
    parser.add_argument("--solver_num_sampling_sequences", type=int, default=16)
    parser.add_argument("--proposer_solver_temperature", type=float, default=1.0)
    parser.add_argument("--num_questions_per_k", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--no_xml_tags_in_prompt", action="store_true")
    parser.add_argument("--no_know_in_rubric", action="store_true")
    parser.add_argument("--scoring_mode", type=str, choices=["pointwise", "pairwise", "pairwise_anchor"], default="pointwise")
    parser.add_argument("--save_interval", type=int, default=64)

    args = parser.parse_args()
    check_args(args)


    if args.use_knowledge and args.existing_questions_path is None:
        knowledge_base = Knowledgebase(args.knowledge_base, seed=args.seed)

    
    llm, tokenizer, proposer_sampling_params, solver_sampling_params, verifier_sampling_params, verifier_scoring_sampling_params = setup_model(args)

    res_path = get_res_path(args)

    res = get_cached_results(res_path)
    
    existing_questions = get_existing_questions(args.existing_questions_path)

    res, start_idx = prefill_knowledge(res, existing_questions, args)    
    

    res_dataset = Dataset.from_list(res)

    completed_dataset = res_dataset.select(range(start_idx))
    to_complete_dataset = res_dataset.select(range(start_idx, len(res)))

    save_interval = args.save_interval

    vllm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:{}/v1".format(args.vllm_port))
    if args.use_teacher_model:
        openai_client = OpenAI()

    for chunk_start_idx in range(start_idx, len(res), save_interval):
        
        chunk_end_idx = min(len(res_dataset),chunk_start_idx+save_interval)
        to_complete_dataset = res_dataset.select(range(chunk_start_idx, chunk_end_idx))
        print("Start generating for chunk {} to {}".format(chunk_start_idx, chunk_end_idx))
        to_complete_dataset = to_complete_dataset.map(
            run_pipeline, 
            num_proc=args.num_proc,
            with_indices=True,
            fn_kwargs={
                "run_args": args,
                "retry_limit": 5,
                "llm": llm,
                "tokenizer": tokenizer,
                "proposer_sampling_params": proposer_sampling_params, 
                "solver_sampling_params": solver_sampling_params, 
                "verifier_sampling_params": verifier_sampling_params, 
                "verifier_scoring_sampling_params": verifier_scoring_sampling_params,
                "existing_questions": existing_questions[chunk_start_idx:chunk_end_idx] if existing_questions is not None else None,
                "num_solver_responses_per_q": int(args.solver_num_sampling_sequences // 4),
            },
            # on_mixed_types="use_json"
        )
        completed_dataset = concatenate_datasets([completed_dataset, to_complete_dataset])
        if os.path.exists(res_path):
            os.remove(res_path)
        completed_dataset.to_json(res_path, lines=True)

    
        valid_proposer_count, valid_solver_count, valid_verifier_count, valid_verifier_scoring_count = get_sizes(completed_dataset)

        print("Total cost from calling the teacher model: {}".format(get_total_costs(completed_dataset)))

        print("valid_proposer_count: {}; valid_solver_count: {}; valid_verifier_count: {}; valid_verifier_scoring_count: {}".format(
            valid_proposer_count, valid_solver_count, valid_verifier_count, valid_verifier_scoring_count
        ))

    


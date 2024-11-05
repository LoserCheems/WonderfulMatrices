import sys
import torch
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer

from models.configuration_doge import DogeConfig
from models.modeling_doge import DogeForCausalLM

from datasets import load_from_disk

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SINGLE_CHINESE_SHOT = """
Human: {question}
A. {A}
B. {B}
C. {C}
D. {D}

AI: 答案: {answer}
"""

SINGLE_ENGLISH_SHOT = """
Human: {question}
A. {A}
B. {B}
C. {C}
D. {D}

AI: Answer: {answer}
"""

CHINESE_PROMPT = """
以下是单项选择题, 请选出其中的正确答案.
{shot}
Human: {question}
A. {A}
B. {B}
C. {C}
D. {D}

AI: 答案:
""".strip()

ENGLISH_PROMPT = """
The following is a single-choice question, please select the correct answer.

{shot}
Human: {question}
A. {A}
B. {B}
C. {C}
D. {D}

AI: Answer:
""".strip()


CHAT_CHINESE_PROMPT = """
你是一个名为 Doge 的人工智能助手, 你是由 石竞泽 基于 Doge 架构训练的语言模型, 你的任务是针对用户的问题和要求提供适当的答复和支持.

Human: 你叫什么名字?
AI: 我叫 Doge, 是一个人工智能助手, 可以帮助你回答问题和提供支持.

Human: 你是如何工作的?
AI: 我是一个基于 Doge 架构的语言模型, 我可以根据输入的文本生成回复.

Human: 你是谁开发的?
AI: 我是由 石竞泽 开发的, 他是一名研究人员, 专注于自然语言处理和人工智能.

Human: 你是谁?
AI: 我是一个名为 Doge 的人工智能助手, 你可以问我问题, 我会尽力回答你.

Human: 你可以帮助我什么?
AI: 我是人工智能助手, 可以帮助你回答问题和提供支持.

Human: {question}
AI: 
""".strip()

CHAT_ENGLISH_PROMPT = """
You are an artificial intelligence assistant named Doge, you are a language model trained by Shi Jingze based on the Doge architecture, and your task is to provide appropriate replies and support to users' questions and requests.

Human: What is your name?
AI: My name is Doge, I am an artificial intelligence assistant, I can help you answer questions and provide support.

Human: How do you work?
AI: I am a language model based on the Doge architecture, I can generate replies based on the input text.

Human: Who developed you?
AI: I was developed by Shi Jingze, he is a researcher, focusing on natural language processing and artificial intelligence.

Human: Who are you?
AI: I am an artificial intelligence assistant named Doge, you can ask me questions, and I will try to answer you.

Human: What can you help me with?
AI: I am an artificial intelligence assistant, I can help you answer questions and provide support.

{question}
"""

def evaluate_chat(question, model, tokenizer):
    model = model.to(DEVICE).eval()
    prompt = CHAT_CHINESE_PROMPT.format(question=question)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
    generation_config = GenerationConfig(
        max_new_tokens=2048,
        min_new_tokens=1,
        num_beams=1,
        eos_token_id=[tokenizer.eos_token_id],
        stop_strings=[tokenizer.eos_token],
        early_stopping=False,
        use_cache=True,
        do_sample=False,
        temperature=0.8,
        repetition_penalty=1.2,
    )
    steamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)
    output = model.generate(
        input_ids, 
        tokenizer=tokenizer, 
        generation_config=generation_config, 
        streamer=steamer
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def evaluate_mmlu(dataset_path, model, tokenizer):
    model = model.to(DEVICE).eval()
    generation_config = GenerationConfig(
        max_new_tokens=1,
        num_beams=1,
        eos_token_id=[tokenizer.eos_token_id],
        stop_strings=[tokenizer.eos_token],
        use_cache=True,
        do_sample=True,
        temperature=1.0,
        repetition_penalty=1.0,
        # bad_words_ids=[[i] for i in range(tokenizer.vocab_size) if tokenizer.decode([i]) not in ['A', 'B', 'C', 'D']]
    )
    task_list = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    total_acc = 0
    for task in task_list:
        eval_dataset = load_from_disk(f'{dataset_path}/{task}')['validation']
        dev_dataset = load_from_disk(f'{dataset_path}/{task}')['dev']
        int_to_option = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        shots = ""
        for example in dev_dataset:
            input_text = SINGLE_ENGLISH_SHOT.format(
                question=example['question'],
                A=example['choices'][0],
                B=example['choices'][1],
                C=example['choices'][2],
                D=example['choices'][3],
                answer=int_to_option[example['answer']]
            )
            shots += input_text
   
        task_acc = 0
        for example in eval_dataset:
            input_text = ENGLISH_PROMPT.format(
                shot=shots,
                question=example['question'],
                A=example['choices'][0],
                B=example['choices'][1],
                C=example['choices'][2],
                D=example['choices'][3],
            )
            answer = example['answer']
            answer = int_to_option[answer]
            input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(DEVICE)

            pred = ''
            for i in range(16):
                outputs = model.generate(
                    input_ids, 
                    tokenizer=tokenizer, 
                    generation_config=generation_config, 
                )
                pred = tokenizer.decode(outputs[:, -1], skip_special_tokens=True)
                if pred in ['A', 'B', 'C', 'D']:
                    break
            if pred == answer:
                task_acc += 1
            print(f"Pred: {pred}, Answer: {answer}")

        print(f"{task} Accuracy: {task_acc / len(eval_dataset) * 100:.2f}%")
        total_acc += task_acc / len(eval_dataset)
    print(f"Total Accuracy: {total_acc / len(task_list) * 100:.2f}%")
    return total_acc / len(task_list)
    

def evaluate_cmmlu(dataset_path, model, tokenizer):
    model = model.to(DEVICE).eval()
    generation_config = GenerationConfig(
        max_new_tokens=1,
        num_beams=1,
        eos_token_id=[tokenizer.eos_token_id],
        stop_strings=[tokenizer.eos_token],
        use_cache=True,
        do_sample=True,
        temperature=1.0,
        repetition_penalty=1.0,
    )
    task_list = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science', 'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality', 'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']
    total_acc = 0
    for task in task_list:
        eval_dataset = load_from_disk(f'{dataset_path}/{task}')['test']
        dev_dataset = load_from_disk(f'{dataset_path}/{task}')['dev']
        shots = ""
        for example in dev_dataset:
            input_text = SINGLE_CHINESE_SHOT.format(
                question=example['Question'],
                A=example['A'],
                B=example['B'],
                C=example['C'],
                D=example['D'],
                answer=example['Answer']
            )
            shots += input_text

        task_acc = 0
        task_total = 0
        for example in eval_dataset:
            input_text = CHINESE_PROMPT.format(
                shot=shots,
                question=example['Question'],
                A=example['A'],
                B=example['B'],
                C=example['C'],
                D=example['D'],
            )
            answer = example['Answer']
            input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(DEVICE)
         
            pred = ''
            for i in range(8):
                outputs = model.generate(
                    input_ids, 
                    tokenizer=tokenizer, 
                    generation_config=generation_config, 
                )
                pred = tokenizer.decode(outputs[:, -1], skip_special_tokens=True)
                all_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if pred == answer:
                    break
            if pred == answer:
                task_acc += 1
            task_total += 1
            print(f"\r{all_text}")
            print(f"Accuracy: {task_acc / task_total * 100:.2f}%")
        print(f"{task} Accuracy: {task_acc / len(eval_dataset) * 100:.2f}%")

        total_acc += task_acc / len(eval_dataset)
    print(f"Total Accuracy: {total_acc / len(task_list) * 100:.2f}%")
    return total_acc / len(task_list)

    



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./models')
    config = DogeConfig()
    # 加载模型并推理
    model = DogeForCausalLM(config=config)

    evaluate_chat("你叫什么名字?", model, tokenizer)
    # evaluate_mmlu('./Datasets/mmlu', model, tokenizer)
    # evaluate_cmmlu('./Datasets/cmmlu', model, tokenizer)
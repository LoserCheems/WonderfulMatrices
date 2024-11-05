from transformers import AutoTokenizer
from datasets import load_from_disk
from argparse import ArgumentParser


def process_cosmopedia(prompt, text):
    prompt = f"[INST]{prompt}[/INST]"
    return tokenizer(prompt, text, padding='max_length', truncation=True, max_length=args.max_length)

def process_chinese_cosmopedia(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=args.max_length)

def process_python_edu(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=args.max_length)

def process_infinity_instruct(conversations):
    prompt = f"[INST]You are an AI assistant named `Doge`, you are a language model trained by `Shi Jingze` based on the `Doge` architecture, and your task is to provide appropriate replies and support to users based on their questions and requests.\n你是一个名为 `Doge` 的人工智能助手, 你是由 `石竞泽` 基于 `Doge` 架构训练的语言模型, 你的任务是针对用户的问题和要求提供适当的答复和支持.\n[/INST][INST]{conversations[0]['value']}[/INST]"
    text = f"{conversations[1]['value']}"
    return tokenizer(prompt, text, padding='max_length', truncation=True, max_length=args.max_length)


argparser = ArgumentParser()
argparser.add_argument("--datasets_dir", type=str, default=None)
argparser.add_argument("--save_dir", type=str, default=None)
argparser.add_argument("--tokenizer_path", type=str, default=None)
argparser.add_argument("--max_length", type=int, default=2048)
argparser.add_argument("--num_proc", type=int, default=1)
args = argparser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, add_eos_token=True)

if __name__ == '__main__':

    # 分词宇宙百科
    # Tokenize Cosmopedia
    dataset = load_from_disk(args.datasets_dir + '/cosmopedia-v2')
    dataset = dataset.map(process_cosmopedia, input_columns=['prompt', 'text'], remove_columns=['prompt', 'text', 'token_length', 'audience', 'format', 'seed_data'], num_proc=args.num_proc)
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/cosmopedia-v2_tokenized')


    # # 分词中文宇宙百科
    # # Tokenize Chinese Cosmopedia
    # dataset = load_from_disk(args.datasets_dir + '/chinese-cosmopedia')
    # dataset = dataset.map(process_chinese_cosmopedia, input_columns=['text'], remove_columns=['text', 'score', 'source', 'data_format'], num_proc=args.num_proc)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + '/chinese-cosmopedia_tokenized')

    # # 分词Python教育
    # # Tokenize Python Education
    # dataset = load_from_disk(args.datasets_dir + '/python-edu')
    # dataset = dataset.map(process_python_edu, input_columns=['text'], remove_columns=['text', 'download_success', 'blob_id', 'repo_name', 'path', 'length_bytes', 'score', 'int_score'], num_proc=args.num_proc)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + '/python-edu_tokenized')

    # # 分词无限指令
    # # Tokenize Infinity Instruct
    # dataset = load_from_disk(args.datasets_dir + '/infinity-instruct-0625')
    # dataset = dataset.map(process_infinity_instruct, input_columns=['conversations'], remove_columns=['id', 'conversations', 'label', 'langdetect', 'source'], num_proc=args.num_proc)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + '/infinity-instruct-0625_tokenized')

    # dataset = load_from_disk(args.datasets_dir + '/infinity-instruct-7M')
    # dataset = dataset.map(process_infinity_instruct, input_columns=['conversations'], remove_columns=['id', 'conversations', 'label', 'langdetect', 'source'], num_proc=args.num_proc)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + '/infinity-instruct-7M_tokenized')

    # dataset = load_from_disk(args.datasets_dir + '/infinity-instruct-Gen')
    # dataset = dataset.map(process_infinity_instruct, input_columns=['conversations'], remove_columns=['id', 'conversations', 'label', 'langdetect', 'source'], num_proc=args.num_proc)
    # print(dataset)
    # dataset.save_to_disk(args.save_dir + '/infinity-instruct-Gen_tokenized')
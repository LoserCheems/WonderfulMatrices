from transformers import AutoTokenizer
from datasets import load_from_disk
from argparse import ArgumentParser


def process_fineweb_edu(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=args.max_length)

def process_cosmopedia(prompt, text):
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=True, padding='max_length', truncation=True, max_length=args.max_length, return_dict=True)

def process_python_edu(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=args.max_length)

def process_open_web_math(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=args.max_length)


argparser = ArgumentParser()
argparser.add_argument("--datasets_dir", type=str, default=None)
argparser.add_argument("--save_dir", type=str, default=None)
argparser.add_argument("--tokenizer_path", type=str, default=None)
argparser.add_argument("--tokens", type=int, default=100_000_000_000)
argparser.add_argument("--max_length", type=int, default=2048)
argparser.add_argument("--num_proc", type=int, default=1)
args = argparser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

if __name__ == '__main__':

    # 计算fineweb-edu, cosmopedia-v2, python-edu, open-web-math的大小
    # Calculate the size of fineweb-edu, cosmopedia-v2, python-edu, open-web-math
    fineweb_edu_size = int(args.tokens * 0.7 // (args.max_length // 1000 * 1000))
    cosmopedia_v2_size = int(args.tokens * 0.2 // (args.max_length // 1000 * 1000))
    python_edu_size = int(args.tokens * 0.05 // (args.max_length // 1000 * 1000))
    open_web_math_size = int(args.tokens * 0.05 // (args.max_length // 1000 * 1000))


    # 分词fineweb-edu
    # Tokenize fineweb-edu
    dataset = load_from_disk(args.datasets_dir + '/fineweb-edu')
    dataset = dataset.select(range(fineweb_edu_size + (1000 * fineweb_edu_size))).map(process_python_edu, input_columns=['text'], remove_columns=['text', 'id', 'metadata'], num_proc=args.num_proc)
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/fineweb-edu_tokenized')

    # 分词宇宙百科
    # Tokenize Cosmopedia
    dataset = load_from_disk(args.datasets_dir + '/cosmopedia-v2')
    dataset = dataset.select(range(cosmopedia_v2_size + (1000 * cosmopedia_v2_size))).map(process_cosmopedia, input_columns=['prompt', 'text'], remove_columns=['prompt', 'text', 'token_length', 'audience', 'format', 'seed_data'], num_proc=args.num_proc)
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/cosmopedia-v2_tokenized')

    # 分词Python教育
    # Tokenize Python Education
    dataset = load_from_disk(args.datasets_dir + '/python-edu')
    dataset = dataset.select(range(python_edu_size + (1000 * python_edu_size))).map(process_python_edu, input_columns=['text'], remove_columns=['text', 'download_success', 'blob_id', 'repo_name', 'path', 'length_bytes', 'score', 'int_score'], num_proc=args.num_proc)
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/python-edu_tokenized')

    # 分词开放网络数学
    # Tokenize Open Web Math
    dataset = load_from_disk(args.datasets_dir + '/open-web-math')
    dataset = dataset.select(range(open_web_math_size + (1000 * open_web_math_size))).map(process_open_web_math, input_columns=['text'], remove_columns=['text', 'url', 'date', 'metadata'], num_proc=args.num_proc)
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/open-web-math_tokenized')
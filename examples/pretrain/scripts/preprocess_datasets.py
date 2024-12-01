from transformers import AutoTokenizer
from datasets import load_from_disk
from argparse import ArgumentParser


def process_fineweb_edu(example, tokenizer, max_length=2048):
    text = example['text']
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length)

def process_cosmopedia(example, tokenizer, max_length=2048):
    prompt = example['prompt']
    text = example['text']
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
    ]
    return tokenizer.apply_chat_template(
        conversation, 
        tokenize=True, 
        padding='max_length', 
        truncation=True, 
        max_length=max_length, 
        return_dict=True
    )

def process_python_edu(example, tokenizer, max_length=2048):
    text = example['text']
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length)

def process_open_web_math(example, tokenizer, max_length=2048):
    text = example['text']
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length)

def main(args):

    # 计算fineweb-edu, cosmopedia-v2, python-edu, open-web-math的大小
    # Calculate the size of fineweb-edu, cosmopedia-v2, python-edu, open-web-math
    fineweb_edu_ratio, cosmopedia_v2_ratio, python_edu_ratio, open_web_math_ratio = 0.7, 0.2, 0.05, 0.05
    fineweb_edu_size = int(args.tokens * fineweb_edu_ratio // (args.max_length // 1000 * 1000))
    cosmopedia_v2_size = int(args.tokens * cosmopedia_v2_ratio // (args.max_length // 1000 * 1000))
    python_edu_size = int(args.tokens * python_edu_ratio // (args.max_length // 1000 * 1000))
    open_web_math_size = int(args.tokens * open_web_math_ratio // (args.max_length // 1000 * 1000))

    # 加载分词器
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 处理fineweb-edu
    # Process fineweb-edu
    dataset = load_from_disk(args.datasets_dir + '/fineweb-edu')
    column_names = dataset.column_names
    dataset = dataset.shuffle(seed=233)
    dataset = dataset.select(
        range(fineweb_edu_size + int((1000 * fineweb_edu_ratio)))
    ).map(
        process_fineweb_edu, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Processing fineweb-edu"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/fineweb-edu_processed')

    # 处理宇宙百科-v2
    # Process Cosmopedia-v2
    dataset = load_from_disk(args.datasets_dir + '/cosmopedia-v2')
    column_names = dataset.column_names
    dataset = dataset.shuffle(seed=233)
    dataset = dataset.select(
        range(cosmopedia_v2_size + int((1000 * cosmopedia_v2_ratio)))
    ).map(
        process_cosmopedia, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Processing cosmopedia-v2"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/cosmopedia-v2_processed')

    # 处理Python教育
    # Process Python Education
    dataset = load_from_disk(args.datasets_dir + '/python-edu')
    column_names = dataset.column_names
    dataset = dataset.shuffle(seed=233)
    dataset = dataset.select(
        range(python_edu_size + int((1000 * python_edu_ratio)))
    ).map(
        process_python_edu, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Processing python-edu"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/python-edu_processed')

    # 处理开放网络数学
    # Process Open Web Math
    dataset = load_from_disk(args.datasets_dir + '/open-web-math')
    column_names = dataset.column_names
    dataset = dataset.shuffle(seed=233)
    dataset = dataset.select(
        range(open_web_math_size + int((1000 * open_web_math_ratio)))
    ).map(
        process_open_web_math, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_length': args.max_length
        },
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Processing open-web-math"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/open-web-math_processed')

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_path", type=str, default="./examples/tokenizer")
    argparser.add_argument("--tokens", type=int, default=100_000_000_000)
    argparser.add_argument("--max_length", type=int, default=2048)
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)

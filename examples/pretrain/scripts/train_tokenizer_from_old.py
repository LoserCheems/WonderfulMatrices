from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
from argparse import ArgumentParser


def example_fineweb_edu(example):
    return {"examples": example["text"]}

def example_cosmopedia_v2(example):
    return {"examples": example["prompt"] + "\n" + example["text"]}

def example_python_edu(example):
    return {"examples": example["text"]}

def example_open_web_math(example):
    return {"examples": example["text"]}

def main(args):

    # 计算fineweb-edu, cosmopedia-v2, python-edu, open-web-math的大小
    # Calculate the size of fineweb-edu, cosmopedia-v2, python-edu, open-web-math
    fineweb_edu_ratio, cosmopedia_v2_ratio, python_edu_ratio, open_web_math_ratio = 0.7, 0.2, 0.05, 0.05
    fineweb_edu_size = int(args.num_examples * fineweb_edu_ratio)
    cosmopedia_v2_size = int(args.num_examples * cosmopedia_v2_ratio)
    python_edu_size = int(args.num_examples * python_edu_ratio)
    open_web_math_size = int(args.num_examples * open_web_math_ratio)

    # 采样 fineweb-edu
    # Sample fineweb-edu
    fineweb_edu = load_from_disk(args.datasets_dir + '/fineweb-edu')
    fineweb_edu = fineweb_edu.shuffle(seed=233)
    column_names = fineweb_edu.column_names
    fineweb_edu = fineweb_edu.select(
        range(fineweb_edu_size)
    ).map(
        example_fineweb_edu, 
        fn_kwargs={},
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Sampling fineweb-edu"
    )

    
    # 采样 cosmopedia-v2
    # Sample cosmopedia-v2
    cosmopedia_v2 = load_from_disk(args.datasets_dir + '/cosmopedia-v2')
    cosmopedia_v2 = cosmopedia_v2.shuffle(seed=233)
    column_names = cosmopedia_v2.column_names
    cosmopedia_v2 = cosmopedia_v2.select(
        range(cosmopedia_v2_size)
    ).map(
        example_cosmopedia_v2, 
        fn_kwargs={},
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Sampling cosmopedia-v2"
    )

    # 采样 python-edu
    # Sample python-edu
    python_edu = load_from_disk(args.datasets_dir + '/python-edu')
    python_edu = python_edu.shuffle(seed=233)
    column_names = python_edu.column_names
    python_edu = python_edu.select(
        range(python_edu_size)
    ).map(
        example_python_edu, 
        fn_kwargs={},
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Sampling python-edu"
    )

    # 采样 open-web-math
    # Sample open-web-math
    open_web_math = load_from_disk(args.datasets_dir + '/open-web-math')
    open_web_math = open_web_math.shuffle(seed=233)
    column_names = open_web_math.column_names
    open_web_math = open_web_math.select(
        range(open_web_math_size)
    ).map(
        example_open_web_math, 
        fn_kwargs={},
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Sampling open-web-math"
    )

    # 合并样本
    # Concatenate samples
    dataset = concatenate_datasets([fineweb_edu, cosmopedia_v2, python_edu, open_web_math])
    dataset.save_to_disk(args.datasets_dir + './datasets/tokenizer_examples')

    dataset = load_from_disk(args.datasets_dir + '/tokenizer_examples')
    dataset = dataset.select(range(args.num_examples))
   

    def get_training_corpus():
        for i in range(0, len(dataset), 1):
            samples = dataset[i : i + 1]["examples"]
            yield samples
    
    # 加载分词器
    # Load tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_tokenizer_path)

    # 训练分词器
    # Train tokenizer
    training_corpus = get_training_corpus()
    new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=32768)

    # 保存新分词器
    # Save new tokenizer
    new_tokenizer.save_pretrained(args.new_tokenizer_save_dir)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--old_tokenizer_path", type=str, default="./examples/tokenizer")
    argparser.add_argument("--new_tokenizer_save_dir", type=str, default="./examples/tokenizer_new")
    argparser.add_argument("--num_examples", type=int, default=1_000_000)
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)

from transformers import AutoTokenizer
from datasets import load_from_disk
from argparse import ArgumentParser


def process_smoltalk(example, tokenizer):
    messages = example['messages']
    example['text'] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )
    return example

def main(args):
    dataset = load_from_disk(args.datasets_dir + '/smoltalk')
    # 保留原始数据集的列名, 以便后续删除多余列
    # Keep the column names of the original dataset for later removal of redundant columns
    columns = dataset['train'].column_names
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset = dataset.map(
        process_smoltalk, 
        fn_kwargs={
            'tokenizer': tokenizer
        },
        num_proc=args.num_proc,
        remove_columns=columns,
        batched=True,
        desc="Applying chat template"
    )
    print(dataset)
    dataset.save_to_disk(args.save_dir + '/finetune_dataset')

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_path", type=str, default="./examples/tokenizer")
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)

from datasets import concatenate_datasets, load_from_disk, Dataset
from argparse import ArgumentParser


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default=None)
    argparser.add_argument("--save_dir", type=str, default=None)
    argparser.add_argument("--num_proc", type=int, default=1)
    args = argparser.parse_args()

    # 合并预训练数据集
    # Concatenate pretraining datasets
    dataset : Dataset = concatenate_datasets([
        load_from_disk(args.datasets_dir + '/fineweb-edu_tokenized'),
        load_from_disk(args.datasets_dir + '/cosmopedia-v2_tokenized'),
        load_from_disk(args.datasets_dir + '/python-edu_tokenized'),
        load_from_disk(args.datasets_dir + '/open-web-math_tokenized')
    ])
    
    # 拆分训练集与测试集并打乱
    # Split train and test sets and shuffle
    dataset = dataset.train_test_split(test_size=1_000, shuffle=True, seed=233)

    # 保存数据集
    # Save dataset
    dataset.save_to_disk(args.save_dir + "/pretrain_datasets", num_proc=args.num_proc)
from datasets import concatenate_datasets, load_from_disk, Dataset
from argparse import ArgumentParser

def main(args):

    # 合并预训练数据集
    # Concatenate pretraining datasets
    dataset : Dataset = concatenate_datasets([
        load_from_disk(args.datasets_dir + '/fineweb-edu_processed'),
        load_from_disk(args.datasets_dir + '/cosmopedia-v2_processed'),
        load_from_disk(args.datasets_dir + '/python-edu_processed'),
        load_from_disk(args.datasets_dir + '/open-web-math_processed')
    ])

    # 拆分训练集与测试集并打乱
    # Split train and test sets and shuffle
    dataset = dataset.train_test_split(test_size=1_000, shuffle=True, seed=233)

    # 保存数据集
    # Save dataset
    dataset.save_to_disk(args.save_dir + "/pretrain_dataset", num_proc=args.num_proc, num_shards={'train': 1024, 'test': 1 })


if __name__ == '__main__':

    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--num_proc", type=int, default=8)
    args = argparser.parse_args()

    main(args)
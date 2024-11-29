from datasets import load_dataset
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    # 下载 smoltalk 数据集
    # Download smoltalk dataset
    dataset = load_dataset("HuggingFaceTB/smoltalk", "all", num_proc=args.num_proc, cache_dir=args.cache_dir)
    print(dataset)
    dataset.save_to_disk(args.save_dir + "/smoltalk", num_proc=args.num_proc)

    # 你还可以下载其他数据集
    # You can also download other datasets
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.deconvolute import deconvolute
from methylbert.trainer import MethylBertFinetuneTrainer
from torch.utils.data import DataLoader

import pandas as pd

import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--train", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--cores", type=int, default=52)
    args = parser.parse_args()

    return args.input, args.train, args.model, args.output, args.cores


if __name__ == "__main__":
    # Parse user input
    input_path, train, model_dir, out_dir, n_cores = parse_arguments()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    tokenizer = MethylVocab(k=3)
    dataset = MethylBertFinetuneDataset(input_path, tokenizer, seq_len=150)
    data_loader = DataLoader(dataset, batch_size=128, num_workers=n_cores)
    df_train = pd.read_csv(train, sep="\t")

    trainer = MethylBertFinetuneTrainer(
        len(tokenizer),
        train_dataloader=data_loader,
        test_dataloader=data_loader,
    )
    trainer.load(model_dir)

    deconvolute(
            trainer=trainer,
            tokenizer=tokenizer,
            data_loader=data_loader,
            output_path=out_dir,
            df_train=df_train,
            adjustment=True,
    )

    print("Deconvolution done!")

from methylbert.data import finetune_data_generate as fdg
from methylbert.data.dataset import MethylBertFinetuneDataset
from methylbert.data.vocab import MethylVocab
from methylbert.trainer import MethylBertFinetuneTrainer
from methylbert.deconvolute import deconvolute

import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--step", type=int, default=4000)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--cores", type=int, default=64)
    args = parser.parse_args()

    return (args.input, args.model, args.output, args.step, args.bs, args.cores)


def load_data(train_dataset: str, test_dataset: str, batch_size: int, num_workers: int):
    tokenizer = MethylVocab(k=3)

    # Load data sets
    train_dataset = MethylBertFinetuneDataset(train_dataset, tokenizer, seq_len=150)
    test_dataset = MethylBertFinetuneDataset(test_dataset, tokenizer, seq_len=150)

    # Create a data loader
    print("Creating Dataloader")

    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=batch_size,
                                   num_workers=num_workers, 
                                   pin_memory=False, 
                                   shuffle=True)

    test_data_loader = DataLoader(test_dataset, 
                                  batch_size=batch_size,
                                  num_workers=num_workers, 
                                  pin_memory=True,
                                  shuffle=False) if test_dataset is not None else None

    return tokenizer, train_data_loader, test_data_loader


def finetune(tokenizer: MethylVocab,
             save_path: str,
             train_data_loader: DataLoader,
             test_data_loader: DataLoader,
             pretrain_model: str,
             steps: int):
    trainer = MethylBertFinetuneTrainer(vocab_size=len(tokenizer),
                                        save_path=save_path + "/bert.model/",
                                        train_dataloader=train_data_loader,
                                        test_dataloader=test_data_loader,
                                        with_cuda=True,
                                        lr=4e-4,
                                        max_grad_norm=1.0,
                                        gradient_accumulation_steps=20,
                                        warmup_step=100,
                                        decrease_steps=200,
                                        eval_freq=1000,
                                        beta=(0.9,0.98),
                                        weight_decay=0.1)
    trainer.load(pretrain_model)
    trainer.train(steps)

    # check output
    assert os.path.exists(os.path.join(save_path, "bert.model/config.json"))
    assert os.path.exists(os.path.join(save_path, "bert.model/dmr_encoder.pickle"))
    assert os.path.exists(os.path.join(save_path, "bert.model/pytorch_model.bin"))
    assert os.path.exists(os.path.join(save_path, "bert.model/read_classification_model.pickle"))

    print("Finetuning done!")


def plot(train_df, eval_df, output_dir):
    """
    Plots the learning curves.

    Args:
    - train_df (DataFrame): DataFrame with training data.
    - eval_df (DataFrame): DataFrame with evaluation data.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))

    # Plot for loss
    ax[0].plot(train_df['step'], train_df['loss'], label='Training Loss')
    ax[0].plot(eval_df['step'], eval_df['loss'], label='Test Loss')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Test Loss')
    ax[0].legend()

    # Plot for ctype_acc
    ax[1].plot(train_df['step'], train_df['ctype_acc'], label='Training Accuracy')
    ax[1].plot(eval_df['step'], eval_df['ctype_acc'], label='Test Accuracy')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Test Accuracy')
    ax[1].legend()

    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    

def main():
    # Parse input parameters
    (input_dir, pre_model, out_dir, n_steps, batch_size, n_cores) = parse_arguments()

    out_finetune = os.path.join(out_dir, "1.finetune")
    out_plot = os.path.join(out_dir, "2.plot")
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    if not os.path.exists(out_finetune): os.mkdir(out_finetune)
    if not os.path.exists(out_plot): os.mkdir(out_plot)

    # 1. Finetune the model
    tokenizer, train_data_loader, test_data_loader = \
        load_data(train_dataset=os.path.join(input_dir, "train_seq.csv"),
                  test_dataset=os.path.join(input_dir, "test_seq.csv"),
                  batch_size=batch_size,
                  num_workers=n_cores)

    finetune(tokenizer, out_finetune, train_data_loader, test_data_loader, pre_model, n_steps)

    # 2. Plot learning acc
    train_df = pd.read_csv(os.path.join(out_finetune, "bert.model/train.csv"), sep='\t')
    eval_df = pd.read_csv(os.path.join(out_finetune, "bert.model/eval.csv"), sep='\t')

    plot(train_df, eval_df, out_plot)

if __name__ == "__main__":
    main()

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
import re
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--step", type=int, default=4000)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--cores", type=int, default=64)
    parser.add_argument("--savefreq", type=int, default=500)
    parser.add_argument("--logfreq", type=int, default=10)
    parser.add_argument("--evalfreq", type=int, default=100)
    parser.add_argument("--logpath", type=str)
    args = parser.parse_args()

    return (args.input, args.model, args.output, args.step, args.bs, args.cores, args.savefreq, args.logfreq, args.evalfreq, args.logpath)


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
             steps: int, 
             save_freq: int, 
             log_freq: int, 
             eval_freq: int):
    trainer = MethylBertFinetuneTrainer(vocab_size=len(tokenizer),
                                        save_path=save_path + "/bert.model/",
                                        train_dataloader=train_data_loader,
                                        test_dataloader=test_data_loader,
                                        with_cuda=True,
                                        lr=1e-5,
                                        max_grad_norm=1.0,
                                        gradient_accumulation_steps=1,
                                        warmup_step=100,
                                        decrease_steps=200,
                                        save_freq=save_freq,
                                        log_freq=log_freq,
                                        eval_freq=eval_freq,
                                        beta=(0.9,0.98),
                                        weight_decay=0.1)
    trainer.load(pretrain_model)
    trainer.train(steps)

    print("Finetuning done!")


# def plot(train_df, eval_df, output_dir):
#     fig, ax = plt.subplots(1, 2, figsize=(16, 16))

#     # Plot for loss
#     ax[0].plot(train_df['step'], train_df['loss'], label='Training Loss')
#     ax[0].plot(eval_df['step'], eval_df['loss'], label='Test Loss')
#     ax[0].set_xlabel('Step')
#     ax[0].set_ylabel('Loss')
#     ax[0].set_title('Training and Test Loss')
#     ax[0].legend()

#     # Plot for ctype_acc
#     ax[1].plot(train_df['step'], train_df['ctype_acc'], label='Training Accuracy')
#     ax[1].plot(eval_df['step'], eval_df['ctype_acc'], label='Test Accuracy')
#     ax[1].set_xlabel('Step')
#     ax[1].set_ylabel('Accuracy')
#     ax[1].set_title('Training and Test Accuracy')
#     ax[1].legend()

#     plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
#     plt.close()


# 基础设置函数
# def set_matplotlib_style():
#     plt.rcParams['figure.figsize'] = (12,8)  # 图形大小
#     plt.rcParams['axes.titlesize'] = 18       # 标题字体大小
#     plt.rcParams['axes.labelsize'] = 15       # x轴和y轴标签字体大小
#     plt.rcParams['xtick.labelsize'] = 12      # x轴刻度字体大小
#     plt.rcParams['ytick.labelsize'] = 12      # y轴刻度字体大小
#     plt.rcParams['legend.fontsize'] = 12      # 图例字体大小
#     plt.rcParams['axes.linewidth'] = 2      # 坐标轴线宽
#     plt.rcParams['xtick.major.size'] = 5      # x轴主刻度大小
#     plt.rcParams['ytick.major.size'] = 5      # y轴主刻度大小
#     plt.rcParams['xtick.major.width'] = 1.5   # x轴主刻度线宽
#     plt.rcParams['ytick.major.width'] = 1.5   # y轴主刻度线宽
#     plt.rcParams['lines.linewidth'] = 2       # 线条宽度
#     plt.rcParams['lines.markersize'] = 8      # 标记大小
#     plt.rcParams['savefig.dpi'] = 300         # 保存图片分辨率
#     plt.rcParams['savefig.format'] = 'pdf'    # 图片保存格式
#     plt.rcParams['grid.alpha'] = 0.6          # 网格线透明度
#     # plt.rcParams['grid.linestyle'] = '--'     # 网格线样式
#     # plt.rcParams['grid.linewidth'] = 0.7      # 网格线宽度
#     plt.rcParams['axes.grid'] = False          # 网格
#     plt.rcParams['axes.edgecolor'] = 'black'  # 边框颜色
#     plt.rcParams['axes.titlepad'] = 15        # 标题与图形之间的距离
#     plt.rcParams['legend.frameon'] = False    # 去掉图例边框
    

# def parse_log_file(file_path):
#     loss_list = []
#     lr_list = []
#     steps = []

#     with open(file_path, 'r') as f:
#         for line in f:
#             match = re.search(r'Train Step (\d+) iter - loss : ([\d.]+) / lr : ([\d.]+)', line)
#             if match:
#                 step = int(match.group(1))
#                 loss = float(match.group(2))
#                 lr = float(match.group(3))
#                 steps.append(step)
#                 loss_list.append(loss)
#                 lr_list.append(lr)
#     return steps, loss_list, lr_list


# def plot_curves(log_path, out_path):
#     set_matplotlib_style()
    
#     steps, loss_list, lr_list = parse_log_file(log_path)

#     # Plot loss curve
#     plt.plot(steps, loss_list, label='Loss')
#     plt.title('Loss Curve')
#     plt.xlabel('Steps')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(os.path.join(out_path, 'loss.pdf'))
#     plt.close()

#     # Plot learning rate curve
#     plt.plot(steps, lr_list, label='Learning Rate', color='orange')
#     plt.title('Learning Rate Curve')
#     plt.xlabel('Steps')
#     plt.ylabel('Learning Rate')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(os.path.join(out_path, 'learning_rates.pdf'))
#     plt.close()

def main():
    # Parse input parameters
    (input_dir, pre_model, out_dir, n_steps, batch_size, n_cores, savefreq, logfreq, evalfreq, log_path) = parse_arguments()

    out_finetune = os.path.join(out_dir, "1.finetune")
    out_plot = os.path.join(out_dir, "2.plot")
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    if not os.path.exists(out_finetune): os.mkdir(out_finetune)
    if not os.path.exists(out_plot): os.mkdir(out_plot)

    # 1. Finetune the model
    tokenizer, train_data_loader, test_data_loader = \
        load_data(train_dataset=os.path.join(input_dir, "train_seq.csv"),
                  test_dataset=os.path.join(input_dir, "val_seq.csv"),
                  batch_size=batch_size,
                  num_workers=n_cores)

    finetune(tokenizer, 
             out_finetune, 
             train_data_loader, 
             test_data_loader, 
             pre_model, 
             n_steps, 
             savefreq, 
             logfreq, 
             evalfreq)

    # 2. Plot loss curve and acc curve
    # train_df = pd.read_csv(os.path.join(out_finetune, "bert.model/train.csv"), sep='\t')
    # eval_df = pd.read_csv(os.path.join(out_finetune, "bert.model/eval.csv"), sep='\t')
    # plot(train_df, eval_df, out_plot)
    # plot_curves(log_path, out_plot)

if __name__ == "__main__":
    main()

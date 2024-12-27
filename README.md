# MethylBERT Fine-Tuning

This document provides instructions for preprocessing and fine-tuning the `MethylBERT` model, a large language model designed for methylation data analysis. The provided scripts include data preprocessing and fine-tuning processes.

## File Structure

- **1.preprocess_data.py**: Preprocesses methylation data for fine-tuning.
- **2.finetune.py**: Fine-tunes the `MethylBERT` model on the preprocessed methylation dataset using a specified batch size and training parameters.
- **3.deconvolute.py**: Uses the finetuned `MethylBERT` model to predict the test data set, assigning probability of each read.

## Requirements
To use the scripts, ensure the following dependencies are installed:

```bash
pip install pandas intervaltree methylbert torch
```

Additional dependencies may be required depending on the dataset format. Check the scripts for any additional libraries used.

## Usage

### Step 1: Data Preprocessing
1. Place your raw methylation data in the expected format (check 1.preprocess_data.py for details).
2. Run the data preprocessing script to generate the input data for fine-tuning.
```bash
python 1.preprocess_data.py --train <path_to_training_data> --test <path_to_test_data> --format <path_to_methylbert_format_file> --dmr <path_to_dmr> --output <path_to_output_directory>
```
This step will clean and format the data as needed for the MethylBERT model.

### Step 2: Fine-Tuning
Once data preprocessing is complete, you can proceed with model fine-tuning.
```bash
python 2.finetune.py --input <path_to_input_directory> --model <path_to_pretraining_model> --output <path_to_output_directory> --step <step> --bs <batch_size> --cores <cores>
```
This script will fine-tune MethylBERT on the methylation dataset, saving the trained model and loss information.

### Step 3: Deconvolution
Deconvolute the test data set after finetuning, report the probability of each read in test set.
```bash
python 3.deconvolute.py --input <path_to_input_csv> --train <path_to_training_csv> --model <path_to_finetuned_model> --output <path_to_output_directory> --cores <cores>
```

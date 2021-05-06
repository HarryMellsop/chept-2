import os
import argparse
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import questionary
from art import tprint

max_game_length = 512
min_game_length = 20

def download_datasets():
    print("Checking the Kingbase Dataset...")
    if os.path.exists('./data/datasets/kingbase-ftfy.txt'):
        print("Kingbase Raw Dataset Found")
    else:
        print("Now downloading the Kingbase dataset...")
        os.system('gsutil cp gs://gpt-2-poetry/data/kingbase-ftfy.txt ./data/datasets/kingbase-ftfy.txt')

    print("Checking the Kaggle Dataset...")
    if os.path.exists('./data/datasets/35-million-chess-games.zip'):
        print("Kaggle Raw Dataset Exists")
    else:
        print("Now downloading the Kaggle dataset...")
        os.system('kaggle datasets download milesh1/35-million-chess-games')
        os.system('mv 35-million-chess-games.zip data/datasets/35-million-chess-games.zip')
        
    print("Checking for Kaggle Unzipped...")
    if os.path.exists('./data/datasets/all_with_filtered_anotations_since1998.txt'):
        print("Kaggle File Extracted")
    else:
        print("Now extracting Kaggle file")
        os.system('unzip ./data/datasets/35-million-chess-games.zip -d ./data/datasets/')


def preprocess_kingbase():
    print("Now processing kingbase-ftfy.txt")

    write_folder = "./data/datasets-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./data/datasets-cleaned/kingbase_cleaned.txt"):
        response = questionary.confirm("It appears that the kingbase file has already been preprocessed; reprocess?").ask()
        if not response:
            return

        os.remove("./data/datasets-cleaned/kingbase_cleaned.txt")

    unprocessed_kingbase_lines = open("./data/datasets/kingbase-ftfy.txt", "r").readlines()

    processed_kingbase_lines = open("./data/datasets-cleaned/kingbase_cleaned.txt", "w")

    line_length = []
    for line in tqdm.tqdm(unprocessed_kingbase_lines):
        split_line = line.split()
        output_line = " ".join(split_line[6:-1]) + "\n"
        output_line = re.sub(r'[0-9]+\.', '', output_line)
        if len(output_line) <= max_game_length and '[' not in output_line and ']' not in output_line:
            processed_kingbase_lines.writelines(output_line)
            line_length.append(len(output_line))

    x = np.array(line_length)

    plt.hist(x, density=True, bins=100)  # density=False would make counts
    plt.ylabel('Relative Frequency')
    plt.xlabel('Sequence Length')
    plt.show()

    print("Total games in the post-processed file: %d", len(line_length))


def preprocess_kaggle():
    print("Now preprocessing all_with_filtered_anotations_since1998.txt")

    write_folder = "./data/datasets-cleaned/"
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # check if this file has already been preprocessed
    if os.path.exists("./data/datasets-cleaned/kaggle_cleaned.txt"):
        response = questionary.confirm("It appears that the kaggle file has already been preprocessed; reprocess?").ask()
        if not response:
            return
        os.remove("./data/datasets-cleaned/kaggle_cleaned.txt")

    unprocessed_kaggle_lines = open("./data/datasets/all_with_filtered_anotations_since1998.txt", "r").readlines()[5:]

    processed_kaggle_lines = open("./data/datasets-cleaned/kaggle_cleaned.txt", "w")

    line_length = []
    for line in tqdm.tqdm(unprocessed_kaggle_lines):
        split_line = line.split()
        for index, token in enumerate(split_line):
            if index % 2 == 0:
                split_line[index] = token[3:]
            else:
                split_line[index] = token[1:]
        output_line = " ".join(split_line[17:]) + "\n"
        if output_line == "\n":
            continue
        output_line = re.sub(r'[0-9]*\.', '', output_line)
        if len(output_line) <= max_game_length and len(output_line) >= min_game_length and '[' not in output_line and ']' not in output_line:
            processed_kaggle_lines.writelines(output_line)
            line_length.append(len(output_line))

    x = np.array(line_length)

    plt.hist(x, density=True, bins=100)  # density=False would make counts
    plt.ylabel('Relative Frequency')
    plt.xlabel('Sequence Length')
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--download', action='store_true')
    parser.add_argument('--preprocess', action='store_true')

    args = parser.parse_args()
    
    if args.download:
        download_datasets()
        
    if args.preprocess:
        tprint("ChePT   Preprocessor")
        try:
            preprocess_kingbase()
        except FileNotFoundError:
            print('Kingbase file not found!')
        try:
            preprocess_kaggle()
        except FileNotFoundError:
            print('Kaggle file not found!')
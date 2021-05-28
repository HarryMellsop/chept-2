import sys
sys.path.insert(1, '.')
from data import vocab
from model import model
from utils import utils
import questionary
import os
import torch

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

def get_prediction(game, gpt_model, stoi, itos, sample=False):
    """
    Extract a prediction from a given model, given a game string as a list of 
    moves that should exist in vocab.
    """

    # if you passed in a str, then we convert it to a list
    if isinstance(game, str):
        game = game.rstrip().split(" ")
    
    # this will print out a warning if we get unk
    x = []
    for move in game:
        try:
            vocab_idx = stoi[move]
        except KeyError:
            print("WARNING: We've got an UNK!")
            print("Game = ", game)
            print("Move = ", move)
            vocab_idx = stoi[BaseVocab().UNK]
        x.append(vocab_idx)

    # x = torch.tensor([stoi.get(c, stoi[BaseVocab().UNK]) for c in x], dtype=torch.long)[None,...].to(device)
    x = torch.tensor(x, dtype=torch.long)[None,...].to(device)

    pred = utils.sample(gpt_model, x, 10, sample=sample)[0]
    # print(pred)
    # print([itos[int(i)] for i in pred])
    return [itos[int(i)] for i in pred][len(game):][0]
    # TODO: MORE TO COME HERE; JUST WANT TO TEST TO SEE WHAT WE'RE GETTING OUT THE OTHER SIDE
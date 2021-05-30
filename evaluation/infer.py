import sys
sys.path.insert(1, '.')
from data.vocab import BaseVocab
from model import model
from utils import utils
import questionary
import os
import torch
from evaluation.evaluate import get_commentary_prediction, get_prediction
from utils.utils import get_recent_ckpt

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # load the GPT model from the parameters
    # ckpt_path = get_recent_ckpt('./ckpts/pretrain-english')
    ckpt_path = './ckpts/ckpts-through-time/finetune_commentary.pt'
    print(f'Loading parameters from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model_config = ckpt['model_config']
    itos = ckpt['itos']
    stoi = ckpt['stoi']

    # build model config
    mconf = model.GPTConfig(
        vocab_size=len(itos),
        args_dict=model_config.__dict__
    )

    # load model weights
    gpt_model = model.GPT(mconf)
    gpt_model = gpt_model.to(device)

    gpt_model.load_state_dict(ckpt['state_dict'])

    # temporary naive prediction logic

    game_str = ''

    while True:
        addor = input(f'Enter your move from state {game_str}')
        game_str += addor

        # get commentary on that first move
        pred = get_commentary_prediction(game_str.split(" ") + [BaseVocab().MASK_CHAR_1], gpt_model, stoi, itos)
        pred = ''.join(pred).replace('_', ' ')
        print(f"Commentary on your move (without sampling): {pred}")

        pred = get_commentary_prediction(game_str.split(" ") + [BaseVocab().MASK_CHAR_1], gpt_model, stoi, itos, sample=True)
        pred = ''.join(pred).replace('_', ' ')
        print(f"Commentary on your move (with sampling): {pred}")

        pred = get_prediction(game_str.split(" "), gpt_model, stoi, itos)
        print(f"My prediction is: {pred}")

        game_str += " " + pred

        pred = get_commentary_prediction(game_str.split(" ") + [BaseVocab().MASK_CHAR_1], gpt_model, stoi, itos)
        pred = ''.join(pred).replace('_', ' ')
        print(f"Commentary on my move (without sampling): {pred}")

        pred = get_commentary_prediction(game_str.split(" ") + [BaseVocab().MASK_CHAR_1], gpt_model, stoi, itos, sample=True)
        pred = ''.join(pred).replace('_', ' ')
        print(f"Commentary on my move (with sampling): {pred}")

        game_str += ' '

    game_str = ''
    commentary_mode = False
    while True:
        addor = input(f"Enter your move from state {game_str}")
        if addor == 'MASKCHAR1':
            addor = BaseVocab().MASK_CHAR_1
            commentary_mode = True
            game_str_pre_addor = game_str
        elif addor == 'MASKCHAR2':
            addor = BaseVocab().MASK_CHAR_2
        game_str += addor
        if not commentary_mode:
            pred = get_prediction(game_str.split(" "), gpt_model, stoi, itos)
            print(f"My prediction is: {pred}")
        else:
            pred = get_commentary_prediction(game_str.split(" "), gpt_model, stoi, itos)
            pred = ''.join(pred).replace('_', ' ')
            print(f"My prediction is (without sampling): {pred}")

            pred = get_commentary_prediction(game_str.split(" "), gpt_model, stoi, itos, sample=True)
            pred = ''.join(pred).replace('_', ' ')
            print(f"My prediction is (with sampling): {pred}")
        
        if not commentary_mode:
            game_str += " " + pred + " "
        if commentary_mode:
            game_str = game_str_pre_addor
        commentary_mode = False
        
        
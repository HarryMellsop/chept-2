from data.vocab import BaseVocab
from model import model
from utils import utils
import questionary
import os
import torch

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Extract a prediction from a given model, given a game string as a list of moves that should exist in vocab
def get_prediction(game, gpt_model, stoi, itos, sample=False):
    x = game
    x = torch.tensor([stoi.get(c, stoi[BaseVocab().UNK]) for c in x], dtype=torch.long)[None,...].to(device)
    pred = utils.sample(gpt_model, x, 10, sample=sample)[0]
    # print(pred)
    # print([itos[int(i)] for i in pred])
    return [itos[int(i)] for i in pred][len(game):][0]
    # TODO: MORE TO COME HERE; JUST WANT TO TEST TO SEE WHAT WE'RE GETTING OUT THE OTHER SIDE

# Get the most recent parameters file from the ckpts
def get_recent_ckpt(ckpt_dir):

    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"Default checkpoint dir at {ckpt_dir} missing!")

    files = os.listdir(ckpt_dir)
    if 'best_loss.pt' in files:
        answer = questionary.confirm("File best_loss.pt found. Use this file?").ask()
        if answer:
            return os.path.join(ckpt_dir, 'best_loss.pt')
    epoch_list = [x for x in files if 'epoch' in x]
    if len(epoch_list) > 0:
        answer = questionary.confirm("Epoch files found. Use best epoch file?").ask()
        if answer:
            epoch_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            return os.path.join(ckpt_dir, epoch_list[0])

    iter_list = [x for x in files if 'iter' in x]
    iter_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)

    return os.path.join(ckpt_dir, iter_list[0])

if __name__ == '__main__':
    # load the GPT model from the parameters
    ckpt_path = get_recent_ckpt('./ckpts/pretrain_default')
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
        game_str += input(f"Enter your move from state {game_str}")
        pred = get_prediction(game_str.split(" "), gpt_model, stoi, itos)
        print(f"My prediction is: {pred}")
        game_str += " " + pred + " "
        
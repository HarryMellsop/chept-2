import sys
sys.path.insert(1, '.')
from data.vocab import BaseVocab
from model import model
from utils import utils
import questionary
import os
import torch
import chess
import chess.engine
import IPython.display as vis

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

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

def check_end(just_moved, board, visualise):
    if visualise: vis.clear_output()
    if visualise: vis.display(board)
    if board.is_stalemate() or board.is_insufficient_material():
        winner = "DRAW"
        if visualise: display("DRAW")
        return (True, winner, 'stalemate')
    
    if board.is_checkmate():
        winner = just_moved
        if visualise: display(f"CHECKMATE, {winner} WINS")
        return (True, winner, 'checkmate')

    return (False, None, None)

def generate_legal_bot_move(board, game_str, gpt_model, stoi, itos):
    illegal_attempts = 0
    for i in range(10):
        # don't bother sampling the first time
        sample = False if i == 0 else True
        bot_move = get_prediction(game_str.split(" "), gpt_model, stoi, itos, sample=sample)

        try:
            board.push_san(bot_move)
            return (bot_move, illegal_attempts)
        except ValueError:
            illegal_attempts += 1
    
    return None, illegal_attempts


def bot_vs_stockfish(gpt_model, itos, stoi, starting_game_str='', visualise=False):
    game_str = starting_game_str

    board = chess.Board()

    for move in game_str.split():
        board.push_san(move)

    if visualise: vis.display(board)
    illegal_moves = []

    while True:
        # let the computer move
        comp_move = engine.play(board, chess.engine.Limit(time=0.05))
        game_str += board.san(comp_move.move)
        board.push(comp_move.move)

        over, winner, reason = check_end('STOCKFISH', board, visualise=visualise)
        if over:
            break

        # let our bot move
        bot_move, illegal_attempts = generate_legal_bot_move(board, game_str, gpt_model, stoi, itos)
        illegal_moves.append(illegal_attempts)
        if bot_move == None:
            if visualise: display(f"CANNOT GENERATE LEGAL MOVE, STOCKFISH WINS")
            if visualise: vis.clear_output()
            if visualise: vis.display(board)
            winner = 'STOCKFISH'
            reason = 'no-legal'
            break
        
        game_str += ' ' + bot_move + ' '
        over, winner, reason = check_end('BOT', board, visualise=visualise)
        if over:
            break

    return illegal_moves, winner, reason, game_str

def initialise_model(ckpt_path):
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

    return gpt_model, stoi, itos

if __name__ == "__main__":
    print("It's time to make a change...")
    gpt_model, stoi, itos = initialise_model('ckpts/pretrain-english/epoch_0_iter_14000.pt')
    while True:
        print(bot_vs_stockfish(gpt_model, itos, stoi))
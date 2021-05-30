import sys

from numpy.lib.function_base import average
sys.path.insert(1, '.')
from data.vocab import BaseVocab
from model import model
from utils import utils
import questionary
import os
import torch
import chess
import chess.engine
import pickle as pickel
import IPython.display as vis
from tqdm import tqdm as mdqt
import numpy as np

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

def get_commentary_prediction(game, gpt_model, stoi, itos, sample=False):
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

    pred = utils.sample(gpt_model, x, 100, sample=sample)[0]
    # print(pred)
    # print([itos[int(i)] for i in pred])
    return [itos[int(i)] for i in pred][len(game):]
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
    for i in range(20):
        # don't bother sampling the first time
        sample = False if i == 0 else True
        bot_move = get_prediction(game_str.split(" "), gpt_model, stoi, itos, sample=sample)

        try:
            board.push_san(bot_move)
            return (bot_move, illegal_attempts)
        except ValueError:
            illegal_attempts += 1
    
    return None, illegal_attempts


def bot_vs_stockfish(gpt_model, itos, stoi, engine, starting_game_str='', visualise=False):
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

def compute_move_scores(game_str, engine):
    # we are black, crucially...
    moves = game_str.split(' ')
    board = chess.Board()
    bot_scores = []
    comp_scores = []
    for i, move in enumerate(moves):
        if i % 2 == 0:
            # this is a computer move, do not analyse it
            board.push_san(move)
        else:
            # this is our move, we need to provide analysis
            board.push_san(move)
            bot_score_tup = engine.analyse(board, chess.engine.Limit(time=0.1), game='key1')
            board.pop()
            computer_move = engine.play(board, chess.engine.Limit(time=0.05))
            board.push(computer_move.move)
            comp_score_tup = engine.analyse(board, chess.engine.Limit(time=0.1), game='key2')
            board.pop()
            board.push_san(move)

            if not bot_score_tup['score'].is_mate() and not comp_score_tup['score'].is_mate():
                bot_scores.append(bot_score_tup['score'].black().score())
                comp_scores.append(comp_score_tup['score'].black().score())

    bot_scores = np.array(bot_scores)
    comp_scores = np.array(comp_scores)

    # split the games up into early, mid and late-game
    early_game_cutoff = 12
    mid_game_cutoff = 24

    bot_early, comp_early = bot_scores[:early_game_cutoff], comp_scores[:early_game_cutoff]
    bot_mid, comp_mid = bot_scores[early_game_cutoff:mid_game_cutoff], comp_scores[early_game_cutoff:mid_game_cutoff]
    bot_late, comp_late = bot_scores[mid_game_cutoff:], comp_scores[mid_game_cutoff:]

    assert (len(bot_early) + len(bot_mid) + len(bot_late)) == len(bot_scores)
    assert (len(comp_early) + len(comp_mid) + len(comp_late)) == len(comp_scores)

    # compute MPS across full-game and across regions

    full_diff = bot_scores - comp_scores
    early_diff = bot_early - comp_early
    mid_diff = bot_mid - comp_mid
    late_diff = bot_late - comp_late

    norm_full = np.mean(np.abs(comp_scores))
    norm_early = np.mean(np.abs(comp_early))
    norm_mid = np.mean(np.abs(comp_mid))
    norm_late = np.mean(np.abs(comp_late))

    MPS_full = np.mean(full_diff / norm_full)
    MPS_early = np.mean(early_diff / norm_early)
    MPS_mid = np.mean(mid_diff / norm_mid)
    MPS_late = np.mean(late_diff / norm_late)

    return (bot_scores, comp_scores, MPS_full, MPS_early, MPS_mid, MPS_late)



def analyse(results_vec, engine):
    # we want to know when the first illegal move occured ever
    # when the first illegal move occured on average
    # number of wins (for our bot)
    # number of stalemates (for our bot)
    # number of losses (for our bot)
    # number of resignations (for our bot)
    # average game length
    # MPS score for the model (across game stages?)

    analytics = {}
    analytics['total_games'] = len(results_vec)

    # illegal move calculations
    illegal_moves_list = [vec[0] for vec in results_vec]
    first_illegal_moves = [np.where(np.array(vec) > 0)[0].min() if np.where(np.array(vec) > 0)[0].size > 0 else -1 for vec in illegal_moves_list]
    analytics['first_illegal_move'] = min([entry for entry in first_illegal_moves if entry != -1])
    analytics['average_first_illegal_move'] = np.mean([entry for entry in first_illegal_moves if entry != -1])

    # game outcomes
    analytics['num_wins'] = sum([1 if vec[1] == 'BOT' else 0 for vec in results_vec])
    analytics['num_losses'] = sum([1 if vec[1] == 'STOCKFISH' else 0 for vec in results_vec])
    analytics['num_stalemates'] = sum([1 if vec[1] == 'DRAW' else 0 for vec in results_vec])

    # resignations
    analytics['resignations'] = sum([1 if vec[2] == 'no-legal' else 0 for vec in results_vec])
    analytics['checkmates'] = sum([1 if vec[2] == 'checkmate' else 0 for vec in results_vec])

    # average game length (moves generated by our bot)
    analytics['avg_game_length'] = np.mean([len(vec[0]) for vec in results_vec])
    analytics['max_game_length'] = np.max([len(vec[0]) for vec in results_vec])
    analytics['min_game_length'] = np.min([len(vec[0]) for vec in results_vec])
    analytics['median_game_length'] = np.median([len(vec[0]) for vec in results_vec])

    # compute the MPS score (across game stages... kms)
    analytics['bot_scores'] = []
    analytics['comp_scores'] = []
    analytics['MPS_full'] = []
    analytics['MPS_early'] = []
    analytics['MPS_mid'] = []
    analytics['MPS_late'] = []
    for game in mdqt(results_vec):
        game_str = game[3]
        bot_scores, comp_scores, MPS_full, MPS_early, MPS_mid, MPS_late = compute_move_scores(game_str, engine)
        analytics['bot_scores'].append(bot_scores)
        analytics['comp_scores'].append(comp_scores)
        analytics['MPS_full'].append(MPS_full)
        analytics['MPS_early'].append(MPS_early)
        analytics['MPS_mid'].append(MPS_mid)
        analytics['MPS_late'].append(MPS_late)

    return analytics

def main():
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    num_trials_per_pt = 2000

    # extract parameters for pretrain chess, finetune chess, pretrain english, finetune commentary

    base_path = './ckpts/ckpts-through-time'
    parameter_paths = ['pretrain_chess.pt', 'finetune_chess.pt', 'pretrain_english.pt', 'finetune_commentary.pt']

    for path in parameter_paths:
        print(f"Analysing {path}")
        ckpt_path = os.path.join(base_path, path)
        gpt_model, stoi, itos = initialise_model(ckpt_path)
        results = []
        for _ in mdqt(range(num_trials_per_pt)):
            results.append(bot_vs_stockfish(gpt_model, itos, stoi, engine=engine))
        
        # perform analytics on the results...
        print("Now computing seriously advanced analytics")
        analytics = analyse(results, engine)
        with open(f'./analytics/anal_{path}.anal', 'wb') as f:
            pickel.dump(analytics, f, pickel.HIGHEST_PROTOCOL)
    
    engine.close()


if __name__ == "__main__":
    main()    
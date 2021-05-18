from data.vocab import AdvancedVocab, CharVocab
import random
import torch
from torch.utils.data import Dataset
import numpy as np

game_splits = {'early': 0,
               'mid': 16,
               'late': 32}


class Pretrain_Word_Level_Chess(Dataset):
    def __init__(self, train_data_path, misc_data_paths, block_size):

        self.block_size = block_size

        # extract all of the relevant training games
        misc_dataset = []
        train_dataset = open(train_data_path, 'r').read().splitlines()
        for path in misc_data_paths:
            misc_dataset = misc_dataset + open(path, 'r').read().splitlines()

        self.vocab = AdvancedVocab(misc_dataset + train_dataset)
        print(f'Data consists of {len(self.vocab.stoi)} unique characters')

        self.data = [game.encode('utf-8').decode('ascii', errors='ignore').strip() for game in train_dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx].split(' ')
        game = game + [self.vocab.PAD_CHAR] * (self.block_size - len(game))

        x = game[:-1]
        y = game[1:]

        x = torch.tensor([self.vocab.stoi.get(c, self.vocab.stoi[self.vocab.UNK]) for c in x], dtype=torch.long)
        y = torch.tensor([self.vocab.stoi.get(c, self.vocab.stoi[self.vocab.UNK]) for c in y], dtype=torch.long)

        return x, y


class Pretrain_Char_Level_Chess(Dataset):

    def __init__(self, train_data_path, misc_data_paths, block_size):

        self.block_size = block_size

        # extract all of the relevant training games
        misc_dataset = []
        train_dataset = open(train_data_path, 'r').read().splitlines()
        for path in misc_data_paths:
            misc_dataset = misc_dataset + open(path, 'r').read().splitlines()

        self.vocab = CharVocab(misc_dataset + train_dataset)
        print(f'Data consists of {len(self.vocab.stoi)} unique characters')

        self.data = [game.encode('utf-8').decode('ascii', errors='ignore') for game in train_dataset]
        print("Maximum data length:")
        print(max([len(entry) for entry in self.data]))
        print(self.block_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]
        game = game + self.vocab.PAD_CHAR * (self.block_size - len(game))

        x = game[:-1]
        y = game[1:]

        x = torch.tensor([self.vocab.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.vocab.stoi[c] for c in y], dtype=torch.long)

        return x, y


class Finetune_Word_Level_Chess(Dataset):
    def __init__(self, train_data_path, misc_data_paths, block_size):

        self.block_size = block_size
        self.portions = [0, 1, 2]

        # extract all of the relevant training games
        misc_dataset = []
        train_dataset = open(train_data_path, 'r').read().splitlines()
        for path in misc_data_paths:
            misc_dataset = misc_dataset + open(path, 'r').read().splitlines()

        self.vocab = AdvancedVocab(misc_dataset + train_dataset)
        self.MASK_CHAR_1 = self.vocab.MASK_CHAR_1
        self.PAD_CHAR = self.vocab.PAD_CHAR
        print(f'Data consists of {len(self.vocab.stoi)} unique characters')

        self.data = [game.encode('utf-8').decode('ascii', errors='ignore').strip() for game in train_dataset]
        self.length = len(self.data)
        # breakpoint()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx].split(' ')
        third = len(game) // 3

        choice = np.random.choice(self.portions)
        min_pt, max_pt = choice * third, (choice + 1) * third

        index = random.randint(min_pt, max_pt)
        x = game[:index] + [self.MASK_CHAR_1] + game[index:index + 1] + [self.MASK_CHAR_1]
        x = x + [self.PAD_CHAR] * (self.block_size - len(x))
        y = [self.PAD_CHAR] * index + [self.MASK_CHAR_1] + game[index:index + 1] + [self.MASK_CHAR_1]
        y = y + [self.PAD_CHAR] * (self.block_size - len(y))

        assert len(x) == len(y) == self.block_size

        x = x[:-1]
        y = y[1:]

        x = torch.tensor([self.vocab.stoi.get(c, self.vocab.stoi[self.vocab.UNK]) for c in x], dtype=torch.long)
        y = torch.tensor([self.vocab.stoi.get(c, self.vocab.stoi[self.vocab.UNK]) for c in y], dtype=torch.long)

        return x, y


class Finetune_Middle(Dataset):

    def __init__(self, data, block_size, pretrain_vocab):

        assert pretrain_vocab, "Must have pretrain vocab for finetuning"

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR_1 = u"\u2047"
        self.MASK_CHAR_2 = u"\u2047"

        chars = list(sorted(list(set(data))))
        if '\n' in chars:
            chars.remove('\n')

        # Check and insert pad and mask chars
        if self.PAD_CHAR in chars:
            chars.remove(self.PAD_CHAR)
        chars.insert(0, self.PAD_CHAR)
        if self.MASK_CHAR_1 in chars:
            chars.remove(self.MASK_CHAR_1)
        chars.insert(0, self.MASK_CHAR_1)
        if self.MASK_CHAR_2 in chars:
            chars.remove(self.MASK_CHAR_2)
        chars.insert(0, self.MASK_CHAR_2)

        self.stoi = pretrain_vocab['stoi']
        self.itos = pretrain_vocab['itos']

        assert len(self.stoi) == len(self.itos)

        breakpoint()
        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        print('Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))[:-1]

        self.data = list(filter(lambda x: len(x.split(' ')) - 1 >= game_splits['mid'], self.data))
        self.min_pt = game_splits['mid']
        self.max_pt = game_splits['late']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]
        spaces = [idx for idx, cur in enumerate(game) if cur == ' ']
        n_spaces = len(spaces)

        index = random.randint(self.min_pt, self.max_pt)
        m_start, m_stop = spaces[index] + 1, spaces[index + 1]
        x = game[:m_start] + self.MASK_CHAR_1 + game[m_start:m_stop + 1] + self.MASK_CHAR_1
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * m_start + self.MASK_CHAR_1 + game[m_start:m_stop + 1] + self.MASK_CHAR_1
        y = y + self.PAD_CHAR * (self.block_size - len(y))

        assert len(x) == len(y) == self.block_size

        x = x[:-1]
        y = y[1:]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y


class Commentary_Dataset(Dataset):

    def __init__(self, data, block_size, pretrain_vocab=None):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR_1 = u"\u2047"
        self.MASK_CHAR_2 = u"\u2048"

        chars = list(sorted(list(set(data))))
        if '\n' in chars:
            chars.remove('\n')

        # Check and insert pad and mask chars
        if self.PAD_CHAR in chars:
            chars.remove(self.PAD_CHAR)
        chars.insert(0, self.PAD_CHAR)
        if self.MASK_CHAR_1 in chars:
            chars.remove(self.MASK_CHAR_1)
        chars.insert(0, self.MASK_CHAR_1)
        if self.MASK_CHAR_2 in chars:
            chars.remove(self.MASK_CHAR_2)
        chars.insert(0, self.MASK_CHAR_2)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        assert len(self.stoi) == len(self.itos)

        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        print('Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))

        self.data = data.split('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]

        idx = game.find(self.MASK_CHAR_1)
        game = game.replace(self.MASK_CHAR_1, self.MASK_CHAR_2)
        x = game + self.PAD_CHAR * (self.block_size - len(game))
        y = self.PAD_CHAR * idx + x[idx:]

        assert len(x) == len(y) == self.block_size

        x = x[:-1]
        y = y[1:]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y


class Directory:

    def __init__(self, func, task, word_level, train_data_path, misc_data_paths, config_args):

        self.func = func
        self.task = task
        self.word_level = word_level
        self.train_data_path = train_data_path
        self.misc_data_paths = misc_data_paths
        self.config_args = config_args

        self.direct = finetune_versions

    def __call__(self):

        return self.direct[(self.func, self.task, self.word_level)](self.train_data_path, self.misc_data_paths, self.config_args['block_size'])


class PretrainDataset(Dataset):

    def __init__(self, data,
                 block_size=1024,
                 pretrain_vocab=None):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR_1 = u"\u2047"
        self.MASK_CHAR_2 = u"\u2047"

        chars = list(sorted(list(set(data))))
        if '\n' in chars:
            chars.remove('\n')

        self.stoi = pretrain_vocab['stoi']
        self.itos = pretrain_vocab['itos']

        assert len(self.stoi) == len(self.itos)

        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        # Check and insert pad and mask chars
        if self.PAD_CHAR in chars:
            chars.remove(self.PAD_CHAR)
        chars.insert(0, self.PAD_CHAR)
        if self.MASK_CHAR_1 in chars:
            chars.remove(self.MASK_CHAR_1)
        chars.insert(0, self.MASK_CHAR_1)
        if self.MASK_CHAR_2 in chars:
            chars.remove(self.MASK_CHAR_2)
        chars.insert(0, self.MASK_CHAR_2)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game = self.data[idx]
        game += self.PAD_CHAR * (self.block_size - len(game))

        x = game[:-1]
        y = game[1:]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y


class Finetune_Char_Level_Chess:
    pass


class Pretrain_Word_Level_Commentary:
    pass


finetune_versions = {('pretrain', 'chess', True): Pretrain_Word_Level_Chess,
                     ('pretrain', 'chess', False): Pretrain_Char_Level_Chess,
                     ('finetune', 'chess', True): Finetune_Word_Level_Chess,
                     ('finetune', 'chess', False): Finetune_Char_Level_Chess,
                     ('pretrain', 'commentary', False): Pretrain_Word_Level_Commentary}

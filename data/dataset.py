import random
import torch
from torch.utils.data import Dataset

game_splits = {'early': 0,
               'mid': 16,
               'late': 32}

class Pretrain_Chess(Dataset):

    def __init__(self, data, block_size, pretrain_vocab):

        self.block_size = block_size
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR_1 = u"\u2047"
        self.MASK_CHAR_2 = u"\u2048" # INCLUDE BOTH MASK CHARACTERS IN PRETRAIN TOO

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

        self.stoi = {i:n for n, i in enumerate(chars)}
        self.itos = {n:i for n, i in enumerate(chars)}

        assert len(self.stoi) == len(self.itos)

        # TODO: Vocab needs to be encoded and include chess + commentary together (from the very start)
        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        print('Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))[:-1]
        print("Maximum data length:")
        print(max([len(entry) for entry in self.data]))
        print(self.block_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]
        game = game + self.PAD_CHAR * (self.block_size - len(game))

        x = game[:-1]
        y = game[1:]

        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

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


class Finetune_Early(Dataset):

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

        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        print('Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))[:-1]

        # ensure all game strings are long enough for process
        self.data = list(filter(lambda x: len(x.split(' ')) - 1 >= game_splits['early'], self.data))
        self.min_pt = 0
        self.max_pt = game_splits['mid']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]
        spaces = [idx for idx, cur in enumerate(game) if cur == ' ']
        n_spaces = len(spaces)

        index = random.randint(self.min_pt, self.max_pt - 1)
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


class Finetune_Late(Dataset):

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

        self.vocab_size = len(self.stoi)
        self.data_size = len(data)

        print('Data has %d characters, %d unique.' % (self.data_size, self.vocab_size))

        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))[:-1]
        
        self.data = list(filter(lambda x: len(x.split(' ')) - 1 >= game_splits['late'], self.data))
        self.min_pt = game_splits['late']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        game = self.data[idx]
        spaces = [idx for idx, cur in enumerate(game) if cur == ' ']
        n_spaces = len(spaces)

        max_idx = n_spaces - 1
        index = random.randint(self.min_pt, max_idx)
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

    def __init__(self, data, version, config_args, pretrain_vocab=None):

        self.data = data
        self.version = version
        self.config_args = config_args
        self.pretrain_vocab = pretrain_vocab

        self.direct = finetune_versions
        self.direct.update({None: PretrainDataset})

    def __call__(self):

        return self.direct[self.version](self.data, self.config_args['block_size'], self.pretrain_vocab)


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

class Finetune_Chess(Dataset):

    pass

class Pretrain_Comm(Dataset):

    pass

class Finetune_Comm(Dataset):

    pass


finetune_versions = {0: Pretrain_Chess,
                     1: Finetune_Chess,
                     2: Pretrain_Comm,
                     3: Finetune_Comm,
                     4: Finetune_Early,
                     5: Finetune_Middle,
                     6: Finetune_Late,
                     7: Commentary_Dataset}


if __name__ == '__main__':

    games = open('data/datasets-cleaned/kingbase_cleaned.txt').read()[:1000]
    print(games)
    # ds = PretrainDataset(games)
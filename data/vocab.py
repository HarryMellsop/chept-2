from tqdm import tqdm
import numpy as np
import string
import pickle as pickel
import os
import questionary

class BaseVocab:
    # data should list of lines
    # ['game1', 'game2']
    def __init__(self):
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR_1 = u"\u2047"
        self.MASK_CHAR_2 = u"\u2048"
        self.UNK = u"\u2049"

class CharVocab(BaseVocab):
    def __init__(self, data):
        super().__init__()

        chars = list({char for word in tqdm(data) for char in word})

        if '\n' in chars:
            chars.remove('\n')

        # Insert sentinel characters
        chars.append(self.PAD_CHAR)
        chars.append(self.MASK_CHAR_1)
        chars.append(self.MASK_CHAR_2)

        # Ensure that all the letters and numbers are in there

        # ensure that we have key characters in place (for prediction later on...)
        for letter in list(string.ascii_lowercase):
            chars.append(letter)
        
        for number in range(0, 10):
            chars.append(str(number))

        for punc in list(string.punctuation):
            chars.append(punc)

        # ensure that we have no duplicates
        chars = list(set(chars))

        self.stoi = {i:n for n, i in enumerate(chars)}
        self.itos = {n:i for n, i in enumerate(chars)}
        self.vocab_size = len(self.stoi)

        assert len(self.stoi) == len(self.itos)

class AdvancedVocab(BaseVocab):

    def __init__(self, data):
        super().__init__()

        use_precompiled = False

        if os.path.exists('./data/datasets/vocab_preprocessed.voc'):

            use_precompiled = questionary.confirm('Found a previously compiled vocab.  Would you like to use this?').ask()
            if use_precompiled:
                vocab_in = open('./data/datasets/vocab_preprocessed.voc', 'rb')
                vocab = pickel.load(vocab_in)
                vocab_in.close()

        if not use_precompiled:

            # load in the entire dictionary of Chess moves that we've seen
            vocab = list({word for game in tqdm(data) for word in game.split()})

            if '\n' in vocab:
                vocab.remove('\n')

            # Insert sentinel characters
            vocab.append(self.PAD_CHAR)
            vocab.append(self.MASK_CHAR_1)
            vocab.append(self.MASK_CHAR_2)
            vocab.append(self.UNK)

            # Ensure that all the letters and numbers are in there

            # ensure that we have key characters in place (for prediction later on...)
            for letter in list(string.ascii_lowercase):
                vocab.append(letter)
            
            for number in range(0, 10):
                vocab.append(str(number))

            for punc in list(string.punctuation):
                vocab.append(punc)

            vocab = list(set(vocab))

            vocab_out = open('./data/datasets/vocab_preprocessed.voc', 'wb')
            pickel.dump(vocab, vocab_out)
            vocab_out.close()
        

        self.stoi = {i:n for n, i in enumerate(vocab)}
        self.itos = {n:i for n, i in enumerate(vocab)}
        self.vocab_size = len(self.itos)

        assert len(self.stoi) == len(self.itos)
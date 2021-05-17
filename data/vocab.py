from tqdm import tqdm
import numpy as np
import string

class BaseVocab:
    # data should list of lines
    # ['game1', 'game2']
    def __init__(self, data):
        self.PAD_CHAR = u"\u25A1"
        self.MASK_CHAR_1 = u"\u2047"
        self.MASK_CHAR_2 = u"\u2048"

class CharVocab(BaseVocab):
    def __init__(self, data):
        super.__init__()

        chars = list({char for word in data for char in word})

        if '\n' in chars:
            chars.remove('\n')

        # Insert sentinel characters
        chars.insert(0, self.PAD_CHAR)
        chars.insert(0, self.MASK_CHAR_1)
        chars.insert(0, self.MASK_CHAR_2)

        # Ensure that all the letters and numbers are in there

        # ensure that we have key characters in place (for prediction later on...)
        for letter in list(string.ascii_lowercase):
            chars.insert(letter)
        
        for number in range(0, 10):
            chars.insert(str(number))

        for punc in list(string.punctuation):
            chars.insert(punc)

        # ensure that we have no duplicates
        chars = list(set(chars))

        self.stoi = {i:n for n, i in enumerate(chars)}
        self.itos = {n:i for n, i in enumerate(chars)}
        self.vocab_size = len(self.stoi)

        assert len(self.stoi) == len(self.itos)

class AdvancedVocab(BaseVocab):
    pass
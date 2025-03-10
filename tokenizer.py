class CurriculumTokenizer:
    def __init__(self):
        self.unk_token = 0
        self.vocab = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
            '+': 4,
            '-': 5,
            '*': 6,
            '^': 7,
            '=': 8,
        }
        for i in range(10):
            self.vocab[str(i)] = 10 + i

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, expression):
        tokens = [self.vocab['<SOS>']]
        for char in expression:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<UNK>'])
        tokens.append(self.vocab['<EOS>'])
        return tokens

    def decode(self, tokens):
        return ''.join(self.reverse_vocab[token] for token in tokens)


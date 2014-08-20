from tokenizer import SimpleTokenizer

class UnigramModel():

    def __init__(self, tokenizer=None, stop_words=set()):
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.docs = None

    def fit(self, D):
        X = []
        V = {}
        for idx, d in enumerate(D):
            if self.tokenizer == None:
                self.tokenizer = SimpleTokenizer()
            token_lst = [token for token in self.tokenizer.tokenize(
                d) if token not in self.stop_words]
            x = [0] * len(token_lst)
            for i, w in enumerate(token_lst):
                if w not in V:
                    V[w] = len(V)
                x[i] = V[w]
            X.append(x)
        return X, dict((v, k) for k, v in V.iteritems())

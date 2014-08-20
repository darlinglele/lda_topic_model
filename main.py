import os
from os import listdir
from tokenizer import SimpleTokenizer
from document_model import UnigramModel
from topic_model import LdaModel

if __name__ == '__main__':
    def get_doc_model():
        base_dir = 'data'
        files = [os.path.join(base_dir, f) for f in listdir(base_dir)]
        D = [open(file).read() for file in files]
        stopwords = set(w.strip() for w in open('stopwords.txt').readlines())
        unigram_model = UnigramModel(
            tokenizer=SimpleTokenizer(), stop_words=stopwords)

        return unigram_model.fit(D)

    X, V = get_doc_model()
    lda_model = LdaModel(n_iter=10)
    lda_model.fit(X, V)

    # lda_model.estimate()

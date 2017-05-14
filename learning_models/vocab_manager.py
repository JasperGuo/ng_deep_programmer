# coding=utf8

from . import util


class VocabManager:
    """
    Vocabulary Manager
    """

    PAD_TOKEN = "<P>"
    PAD_TOKEN_ID = 0

    def _read(self, vocab_path):
        data = list()
        with open(vocab_path, "r") as f:
            for line in f:
                data.append(line.strip().replace("\n", ""))
        return data

    def __init__(self, vocab_path):
        self._vocab = self._read(vocab_path)

        self._vocab_id2word = dict()
        self._vocab_word2id = dict()

        _id = 1
        for word in self._vocab:
            self._vocab_id2word[_id] = word
            self._vocab_word2id[word] = _id
            _id += 1

        self._vocab_id2word[self.PAD_TOKEN_ID] = self.PAD_TOKEN
        self._vocab_word2id[self.PAD_TOKEN] = self.PAD_TOKEN_ID

    @property
    def vocab(self):
        return self._vocab

    @property
    def vocab_len(self):
        return len(self._vocab)

    def word2id(self, word):
        return util.get_value(self._vocab_word2id, str(word))

    def id2word(self, wid):
        return util.get_value(self._vocab_id2word, wid)

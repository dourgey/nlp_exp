import os
from typing import List

class Tokenizer:
    def __init__(self, vocab_path: str = r'..\..\datasets\vocab.txt'):
        # there are two vocab file in ..\..\datasets\
        # vocab.txt is the chinese bert vocab
        # vocab_bert_based_uncased.txt is the bert base uncased vocab
        self.token_to_id_dict, self.id_to_token_dict, self.vocab_size = self.load_vocab(vocab_path)

    def load_vocab(self, path: str):
        assert os.path.exists(path)
        vocab = open(path, 'r', encoding='UTF-8').read().split('\n')
        special_tokens = ['[MASK]', '[SEP]', '[CLS]', '[UNK]', '[PAD]']
        for st in special_tokens:
            if st not in vocab:
                vocab.insert(0, st)
        id_to_token_dict = {i: x for i, x in enumerate(vocab)}
        token_to_id_dict = {x: i for i, x in enumerate(vocab)}
        return token_to_id_dict, id_to_token_dict, len(vocab)

    def encode(self, text: str, level='character') -> List[int]:
        assert level in ['character', 'word']
        if level == 'word':
            return self.convert_tokens_to_ids(text.strip().split())
        else:
            return self.convert_tokens_to_ids([x for x in text.strip()])

    def decode(self, token_ids, join_char = '') -> str:
        return join_char.join(self.convert_ids_to_tokens(token_ids))


    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.convert_id_to_token(_id) for _id in ids]

    def convert_id_to_token(self, id: int) -> str:
        assert id < self.vocab_size
        return self.id_to_token_dict[id]

    def convert_token_to_id(self, token: str):
        if token not in self.token_to_id_dict:
            return self.token_to_id_dict['[UNK]']
        return self.token_to_id_dict[token]

    def convert_tokens_to_ids(self, tokens: List[str]):
        return [self.convert_token_to_id(token) for token in tokens]

if __name__ == '__main__':
    tokenizer = Tokenizer(r'..\..\datasets\vocab_bert_based_uncased.txt')

    print(tokenizer.encode('i love you', level='word'))
    print(tokenizer.encode('我爱你', level='character'))

    print(tokenizer.decode([1045, 2293, 2017], ' '))
    print(tokenizer.decode([1045, 2293, 2017]))
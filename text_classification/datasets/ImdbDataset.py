import torch

from src.utils.tokenizer import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import IMDB
from sklearn.model_selection import train_test_split


# train_iter = IMDB(split='train')
# train_iter, test_iter = IMDB()


def data_generate(data_iter, level='word'):
    assert level in ['word', 'char']
    x = []
    y = []
    for label, text in data_iter:
        if level == 'word':
            x.append(text.split())
        else:
            x.append([char for char in text])
        y.append(label)
    y = [1 if l == 'pos' else 0 for l in y]
    return x, y

class IMDBDataset(Dataset):
    def __init__(self, vocab_path, data_root, num_words=None, skip_top=0, maxlen=None, type="train", valid_size=0.2, level='word'):

        self.tokenizer = Tokenizer(vocab_path)
        assert type in ["train", "valid", "test"]

        if type == 'test':
            test_iter = IMDB(split='test', root=data_root)
            x, y = self.data_generate(test_iter, level)

        if type == 'train':
            train_iter = IMDB(split='train', root=data_root)
            x, y = self.data_generate(train_iter, level)
            x, y = x[: int((1 - valid_size) * len(x))], y[: int((1 - valid_size) * len(x))]

        if type == 'valid':
            valid_iter = IMDB(split='train', root=data_root)
            x, y = self.data_generate(valid_iter, level)
            x, y = x[-int(valid_size * len(x)):], y[-int(valid_size * len(x)):]

        self.x = [torch.tensor(self.tokenizer.convert_tokens_to_ids(sentence)) for sentence in x]
        self.y = torch.tensor(y)

    def data_generate(self, data_iter, level='word'):
        # 默认转化为小写
        assert level in ['word', 'char']
        x = []
        y = []
        for label, text in data_iter:
            if level == 'word':
                x.append([word.lower() for word in text.split()])
            else:
                x.append([char.lower() for char in text])
            y.append(label)
        y = [1 if l == 'pos' else 0 for l in y]
        return x, y

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return len(self.x)



if __name__ == '__main__':
    train_set = IMDBDataset(type='train')
    valid_set = IMDBDataset(type='valid')
    test_set = IMDBDataset(type='test')
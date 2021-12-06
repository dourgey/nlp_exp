import torch

def scaled_dot_product_attention(Q, K, V, mask=None):
    """

    :param Q: [batch_size, q_len]
    :param K: [batch_size, seq_len, k_len]
    :param V: [batch_size, seq_len, v_len]
    :param mask: BoolTensor
    :return:
    """
    score = torch.bmm(Q, K.transpose(1, 2))
    if mask is not None:
        score.masked_fill_(mask, float('-inf'))  # mask
    attn = torch.softmax(score, -1) / torch.sqrt(torch.tensor(Q.shape[-1]))
    out = torch.bmm(attn, V)
    return out, attn


if __name__ == '__main__':
    q = torch.randn(3, 5, 7)
    k = torch.randn(3, 5, 7)
    v = torch.randn(3, 5, 7)
    mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool)
    print(scaled_dot_product_attention(q, k, v))
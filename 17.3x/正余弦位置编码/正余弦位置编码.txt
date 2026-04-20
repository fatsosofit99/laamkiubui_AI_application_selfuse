import numpy as np

def token_embedding(token_ids: list, vocab_size: int, embed_dim: int) -> np.ndarray:
    embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    return embedding_table[token_ids]


def sinusoidal_positional_encoding(seq_len: int, embed_dim: int) -> np.ndarray:
    #TODO
    pe = np.zeros((seq_len,embed_dim))
    position=np.arange(0,seq_len)[:,np.newaxis]
    div_term=np.power(10000,np.arange(0,embed_dim,2)/embed_dim)
    pe[:,0::2]=np.sin(position/div_term)
    pe[:,1::2]=np.cos(position/div_term)

    return pe



def add_embedding_and_position(embedding: np.ndarray, position_encoding: np.ndarray) -> np.ndarray:
    #TODO
    return embedding+position_encoding

if __name__ == '__main__':
    np.random.seed(42)
    token_ids = [3, 7, 1, 9]
    vocab_size = 1000
    embed_dim = 4

    token_vecs = token_embedding(token_ids, vocab_size, embed_dim)
    pos_enc = sinusoidal_positional_encoding(len(token_ids), embed_dim)
    final_output = add_embedding_and_position(token_vecs, pos_enc)

    print("Token Embedding:\n", token_vecs)
    print("Positional Encoding:\n", pos_enc)
    print("Final Output:\n", final_output)
# BPE (byte pair encode) algorithm functions
def get_pairs(ids: list[int], pairs: dict) -> dict[tuple[int, int], int]:
    '''
    given a list of tokens, return a dictionary of pairs and their frequency

    ids=[1, 2, 3, 1, 2, 3, 5] -> pairs_freq={(1, 2): 2, (2, 3): 2, (3, 1): 1, (3, 5): 1}
    '''

    for pair in zip(ids, ids[1:]): # [(1, 2), (2, 3), (3, 1), (3, 5)]
        pairs[pair] = pairs.get(pair, 0) + 1

    return pairs # {(1, 2): 2, (2, 3): 2, (3, 1): 1, (3, 5): 1}

def merge(ids: list[int], pair: tuple[int, int], id: int) -> None:
    '''
    given a pair of tokens and a token (id), replace all occurrences of the pair with the token

    ids=[1, 2, 3, 1, 2, 3, 5] pair=(1, 2) id=4 -> new_ids=[4, 3, 4, 3, 5]
    '''

    for i in range(len(ids) - 1):
        curr_pair = tuple(ids[i:i + 2])

        if curr_pair == pair:
            ids[i:i + 2] = [id]
        
    return ids

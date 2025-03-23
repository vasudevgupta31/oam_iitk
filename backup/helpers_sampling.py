import numpy as np



def int_to_smile(array, indices_token, pad_char):
    """
    Converts an array of integers into a list of SMILES strings.
    Removes padding characters.
    """
    return [''.join(indices_token[str(int(x))] for x in seq).replace(pad_char, '') for seq in array]


def one_hot_encode(token_lists, n_chars):
    """
    Converts token indices into one-hot encoding.
    """
    output = np.zeros((len(token_lists), len(token_lists[0]), n_chars), dtype=np.float32)
    rows, cols = np.indices((len(token_lists), len(token_lists[0])))
    output[rows, cols, token_lists] = 1
    return output


def get_token_proba(preds, temp):
    """
    Applies temperature-based sampling on model predictions.
    """
    preds = np.asarray(preds, dtype=np.float64)
    exp_preds = np.exp(preds / temp)  # Normalize in one step
    probas = exp_preds / np.sum(exp_preds)
    
    return probas, np.random.choice(len(probas), p=probas)  # Directly sample


def sample(model, temp, start_char, end_char, max_len, indices_token, token_indices):
    """
    Generates a SMILES sequence using the trained model.
    """
    n_chars = len(indices_token)
    seed_token = [token_indices[start_char]]
    generated = [indices_token[str(seed_token[0])]]

    while generated[-1] != end_char and len(generated) < max_len:
        x_seed = one_hot_encode(np.array(seed_token, dtype=int).reshape(1, -1), n_chars)
        logits = model.predict(x_seed, verbose=0)[0, -1]
        _, next_char_ind = get_token_proba(logits, temp)
        generated.append(indices_token[str(next_char_ind)])
        seed_token.append(next_char_ind)
            
    return ''.join(generated)


def softmax(preds):
    """
    Computes softmax probabilities.
    """
    exp_preds = np.exp(preds - np.max(preds))  # Prevents overflow
    return exp_preds / np.sum(exp_preds)


def optimized_sample(model, temp, start_char, end_char, max_len, indices_token, token_indices):
    """
    Optimized version of the SMILES sequence generation function.
    
    Args:
        model: Trained Keras model
        temp: Temperature for sampling
        start_char: Character to start sequences with
        end_char: Character that marks the end of a sequence
        max_len: Maximum length of generated sequences
        indices_token: Mapping from indices to tokens
        token_indices: Mapping from tokens to indices
        
    Returns:
        str: Generated SMILES string
    """
    n_chars = len(indices_token)
    
    # Pre-convert tokens to avoid repeated conversions
    start_token_id = int(token_indices[start_char])
    end_token_id = int(token_indices[end_char]) if end_char in token_indices else -1
    
    # Pre-allocate token lists with reasonable capacity
    seed_token = np.zeros(max_len, dtype=np.int32)
    seed_token[0] = start_token_id
    token_len = 1
    
    # Store the string representation of end_token_id for comparison
    end_token_str = str(end_token_id)
    
    # Pre-generate one-hot encoding matrix to reuse across iterations
    x_seed = np.zeros((1, token_len, n_chars), dtype=np.float32)
    
    # Cache for indices_token lookups
    token_lookup_cache = {}
    
    # Pre-cache the common tokens to avoid dict lookups
    for i in range(min(100, n_chars)):  # Cache the most common tokens
        token_lookup_cache[str(i)] = indices_token[str(i)]
    
    # Generated characters list
    generated = [indices_token[str(start_token_id)]]
    
    while len(generated) < max_len:
        # Update only the relevant part of the one-hot tensor
        # Resize if needed
        if token_len > x_seed.shape[1]:
            new_x_seed = np.zeros((1, token_len, n_chars), dtype=np.float32)
            new_x_seed[:, :x_seed.shape[1], :] = x_seed
            x_seed = new_x_seed
            
        # Efficient one-hot encoding
        x_seed.fill(0)  # Clear previous data
        for i in range(token_len):
            x_seed[0, i, seed_token[i]] = 1.0
            
        # Get prediction
        logits = model.predict(x_seed[:, :token_len, :], verbose=0)[0, -1]
        
        # Sample next token
        _, next_char_ind = get_token_proba(logits, temp)
        next_char_ind_str = str(next_char_ind)
        
        # Use cache if available, otherwise do the lookup
        if next_char_ind_str in token_lookup_cache:
            next_token = token_lookup_cache[next_char_ind_str]
        else:
            next_token = indices_token[next_char_ind_str]
            token_lookup_cache[next_char_ind_str] = next_token
            
        generated.append(next_token)
        
        # Add to seed token array
        seed_token[token_len] = next_char_ind
        token_len += 1
        
        # Check for end condition
        if next_token == end_char or next_char_ind_str == end_token_str:
            break
            
    # Join all characters efficiently
    return ''.join(generated)

import numpy as np

with open('C:/Users/likhi/Documents/GPT_Andrej/inputs.txt', 'r') as file:
    text = file.read()

chars = set(text)
data_size, vocab_size = len(text), len(chars)
#print(data_size, vocab_size)

chars_to_idx = {char:idx for idx, char in enumerate(chars)}
idx_to_chars = {idx:char for idx, char in enumerate(chars)}

h_dim = 4
sequence_length = 8
learning_rate = 1e-3

h_state = np.zeros((h_dim, 1))

# rnn weights
W_h = np.random.normal(size= (h_dim, h_dim))
W_in = np.random.normal(size= (h_dim, vocab_size))
W_op = np.random.normal(size= (vocab_size, h_dim))
b_h = np.zeros((h_dim, 1))
b_out = np.zeros((vocab_size, 1))

def compute_loss(inputs, targets, h_prev):

    """ both inputs and targets are integer sequences - token indices
    """

    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = h_prev
    loss = 0

    # forward pass
    for t in range(len(inputs)):

        xs = np.zeros((vocab_size,1))
        xs[inputs[t]] = 1

        hs[t] = np.tanh(np.matmul(W_h, hs[t -1]) + np.matmul(W_in, xs[t]) + b_h)
        ys[t] = np.matmul(W_op, hs[t]) + b_out
        ps[t] = np.exp(ys[t])/ np.sum(np.exp(ys[t]))

        loss += -np.sum(targets[t]*np.log(ps[t]))

    # backward pass
    # ??????????

def sample(seed_idx, n, hs):

    result = list()
    x = np.zeros((vocab_size,1))
    x[seed_idx] = 1

    for i in range(n):
        
        hs = np.tanh(np.matmul(W_h, hs) + np.matmul(W_in, x) + b_h)
        y = np.matmul(W_op, hs) + b_out
        p = np.exp(y)/ np.sum(np.exp(y))
        idx = np.random.choice(range(vocab_size), p= p.flatten())

        x = np.zeros((vocab_size,1))
        x[idx] = 1

        result.append(idx)
    
    return result

n, p = 0,0

# memory ????
mW_in = np.zeros_like(W_in)
mW_op = np.zeros_like(W_op)
mW_h = np.zeros_like(W_h)
mb_h = np.zeros_like(b_h)
mb_out = np.zeros_like(b_out)

# this mechanism is used to smooth out noise in loss function and obtain a smooth curve
# additionally, it assumes the model initially selects datapoints using a uniform probability distribution over vocab_size
# we are multiplying by sequence_length cause during training we are calculating loss across each character/ timestep/ sequence_length
smooth_loss = -np.log(1.0/ vocab_size) * sequence_length  

while True:

    if n == 0 or ((p + sequence_length +1) > len(text)):
        p = 0
        h_prev = np.zeros((h_dim,1))
    
    inputs = [chars_to_idx[char] for char in text[p:p+sequence_length]]
    targets = [chars_to_idx[char] for char in text[p+1: p+sequence_length+1]]

    if n%100 == 0:
        tokens = sample(seed_idx= inputs[0], n=200, hs = h_prev)
        answer = "".join([idx_to_chars[int(token)] for token in tokens])
        print(answer)

    loss, dW_in, dW_op, dW_h, db_h, db_out, h_prev = compute_loss(inputs, targets, h_prev)
    smooth_loss = smooth_loss * 0.99 + loss * 0.01
    if n%100 == 0: print("Loss at {n}th epoch is: {loss}")

# this is the update step of Adagrad(Adaptive Gradient)
# Adagrad adjust the learnig rate of each parameter based on the history/ magnitude/ frequency of updates it has been receiving
    for param, dparam, mem in zip([W_in, W_op, W_h, b_h, b_out], 
                                  [dW_in, dW_op, dW_h, db_h, db_out], [mW_in, mW_op, mW_h, mb_h, mb_out]):
        
        # accumulate squared gradients for each parameter - memory
        mem += (dparam*dparam)
        # update step; the dparam/np.sqrt() term scales the gradient of each parameter based on the magnitude of updates it has been receiving
        # it is easy to see that, for a specific weight, if the gradient has been large many times, mem becomes large, and the update becomes smaller.
        param += -learning_rate* (dparam/np.sqrt(mem + 1e-8))
    
    n += 1
    p += sequence_length
    





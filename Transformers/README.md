# Transformers

![image.png](image.png)

# Why Transformers

![image.png](image%201.png)

Context: RNNs were previously used

1. had no long-range dependencies 
    1. i.e. only look at a small amount of previous token, whereby in language, even a token at the start of the sentence can influence the last token in the sentence
2. had sequential processing 
    1. thus no parallelism 
3. **Computational Efficiency**
    1. Sequential Operations:
        1. Self-attention has constant `O(1)` number of sequentially executed operations, while recurrent layer has number of operations linear to sequence length `O(n)` 
        2. Complexity per layer:
            1. in self-attention layer, each element attends to all other elements ⇒ $O(n^2)$
                
                For each interactions, there is a matrix multiplication of dimension d  (dimension of input vectors / embedding) ⇒ $O(n^2)$
                
            2. in recurrent layer, operations mainly matrix muplication betrween input and hidden states ($d^2$ complexity). This operation is executed for each element ⇒ $O(n * d^2)$ 
            - WV∈Rdmodel×dvW_V \in \mathbb{R}^{d_{\text{model}} \times d_v}WV∈Rdmodel×dv
            1. As long as sequence length `n` is smaller than dimension of input vectors `d`, self-attention layers are faster than recurrent layers 

# How Transformers Work

## 1. Word embeddings

![image.png](image%202.png)

convert input / output to word embedding of dimension `d` 

- Same weight matrices are used for embedding for both input and out tokens, and the pre-softmax linear transformation layer in the decoder
- This allows a learned shared set of paramteres for both token embeddings and ffinal token prediction step in the decoder.

## **2. Positional Encoding**

![image.png](image%203.png)

![](https://aiml.com/wp-content/uploads/2023/09/example_of_positional_encoding_in_transformers.png)

RNNs process words sequentially, while transformers process all words at once ⇒ no in-built mechanism to consider word order. 

Add positional encoding within the embeddings, so that transformer knows the position.

For token at position $p$ and dimension $i$:

**Even dimensions:**

$$
\text{PE}(p, 2i) = \sin\left( \frac{p}{10000^{2i/d_{\text{model}}}} \right)
$$

**Odd dimensions:**

$$
\text{PE}(p, 2i+1) = \cos\left( \frac{p}{10000^{2i/d_{\text{model}}}} \right)
$$

Typically add this vector to the token embedding:

$$
x_p = \text{TokenEmbedding}(p) + \text{PositionalEncoding}(p)
$$

Advantages:

- Unique encoding for each position
- All values are of values [-1, 1]
- Position encoding independent of N

## 3. Self-Attention and Multi-Head Attention

![image.png](image%204.png)

![image.png](image%205.png)

![image.png](image%206.png)

For each token in a sequence, we want to measure how relevant the token is to all other tokens in the same sequence, and place more importance of those that are relevant to capture contextual information.

### Self-Attention: From Tokens to Queries, Keys, and Values (Q, K, V)

Assume:

- A sequence of `n` tokens
- Model (embedding) dimension `d_model`

1. Token Embeddings

Each token is embedded into a vector of dimension `d_model`.

Stack the token embeddings into a matrix:

$$
X \in \mathbb{R}^{n \times d_{\text{model}}}
$$

Each row of `X` corresponds to one token embedding.

1. Learned Projection Matrices

The Transformer learns three linear projection matrices:

$$
W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k} \newline W_K \in \mathbb{R}^{d_{\text{model}} \times d_k} \newline W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

1. Computing Queries, Keys, and Values

The projections are computed via matrix multiplication:

$$
Q = X W_Q \in \mathbb{R}^{n \times d_k}
 \newline K = X W_K \in \mathbb{R}^{n \times d_k}\newline V = X W_V \in \mathbb{R}^{n \times d_v}
$$

1. Interpretation
- Each row of `Q` is the **query vector** for a token
- Each row of `K` is the **key vector** for a token
- Each row of `V` is the **value vector** for a token

All three representations are derived from the same input matrix `X`, but via different learned projections.

1. Intuition
- Queries represent *what a token is looking for*
- Keys represent *what a token offers*
- Values represent *the information a token passes on*

These are later combined using attention to determine how tokens interact.

1. Self-Attention Computation 

Attention Score: Each token’s query is compared against all keys using a scaled dot product.

Attention Weights: Row-wise softmax to convert scores into attention weights, where each row of A sums to 1 and represents how attention is distributed over all tokens.

Self-Attention Output / Head: Compute a weighted sum of value vector. Each output row is a context aware representation of a token. 

$$
S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times n} \newline A = \mathrm{softmax}(S) \in \mathbb{R}^{n \times n} \newline Head  = A V \in \mathbb{R}^{n \times d_v}
$$

### Multi-Head Attention

![image.png](image%207.png)

![image.png](image%208.png)

Same as Self-Attention, but split subspaces of Q, K and V matrices into `h` smaller matrices instead of using 1 whole matrix for Q, K and V to get `h` heads. 

$$
\begin{aligned}
&\textbf{Multi-Head Self-Attention} \\[4pt]
&n:\text{ sequence length},\quad d_{\text{model}}:\text{ embedding dimension},\quad h:\text{ number of heads} \\[2pt]
& d_q = d_k = d_v = \frac{d_{\text{model}}}{h} \\[6pt]

&X \in \mathbb{R}^{n \times d_{\text{model}}} \\[6pt]

&\forall i \in \{1,\dots,h\}: \quad
W_Q^{(i)},\, W_K^{(i)},\, W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k} \\[6pt]

&Q^{(i)} = X W_Q^{(i)} \in \mathbb{R}^{n \times d_k} \\[-1pt]
&K^{(i)} = X W_K^{(i)} \in \mathbb{R}^{n \times d_k} \\[-1pt]
&V^{(i)} = X W_V^{(i)} \in \mathbb{R}^{n \times d_v} \\[8pt]

&A^{(i)} = \mathrm{softmax}\!\left( \frac{Q^{(i)} (K^{(i)})^\top}{\sqrt{d_k}} \right)
\in \mathbb{R}^{n \times n} \\[8pt]

&\mathrm{head}^{(i)} = A^{(i)} V^{(i)} \in \mathbb{R}^{n \times d_v} \\[8pt]

&\mathrm{Concat}\!\left(\mathrm{head}^{(1)},\dots,\mathrm{head}^{(h)}\right)
\in \mathbb{R}^{n \times (h d_v)} \\[6pt]

&\mathrm{MultiHeadAttention}(X)
= \mathrm{Concat}\!\left(\mathrm{head}^{(1)},\dots,\mathrm{head}^{(h)}\right) W_O \\[4pt]

&W_O \in \mathbb{R}^{(h d_v) \times d_{\text{model}}}
\quad
\left(
\text{if } d_v=\tfrac{d_{\text{model}}}{h},\;
W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
\right)
\end{aligned}
$$

### Benefits of Multi-Head Attention:

- Parallelism as computation for each head is independent
- Each head specialises on different kind of relationships — note that this is an emergent behaviour.
    - e.g. One head might focus on nearby words, while another on long-range links, and another on punctuation
- More expressive than having one attention pattern
    - Can represent context as a mix of several distinct relationships / patterns
- Small overall cost
    - Parameters after concatenating the heads together will still be the same

## 4. Residual Connection and Layer Normalisation

### Residual Connection

![image.png](image%209.png)

This layer is the `Add & Norm` part 

Residual connection is an arichtectural shorcut that allows information to flow through, by passing one or more layers of neural network computations.

This mechanism addresses the common issue of vanishing gradients in deep network. Gradients diminish as they propagate backward through numerous layers.

Hence, there will be smoother gradient flow ⇒ more effective training 

$$
y = F(x) + x \newline 

$$

Residual Connections help the network train better as the neural network contribute very little at initialisation and the unimpeded 

### Layer Normalisation (LN)

Each token’s hidden vector is normalised (z-score method), across features. This helps to stabilise training and make gradients/updates more consistent across layers 

Post-LN (original Transformer): 

$$
y = LN(x + F(x))
$$

Pre-LN (modern-day LLMs): 

$$
y = x + F(LN(x)) 
$$

Pre-LN is more commonly used because gradient flow is simpler (i.e. lower chance of getting transformed to extreme values).

## 5. FFN

![image.png](image%2010.png)

The normalised residual output is fed into a position-wise feed forward neural network, which  are a couple of layers with `ReLu` activation. The FFN output has residual connection and LM again.

## 6. Cross-Attention

![image.png](image%2011.png)

Self-attention occurs within the encoder and decoder itself, where tokens receive context of tokens in the same sequence. 

Cross-attention occurs between the encoder and decoder, where the K and V matrices are from the encoder, while Q matrix comes from the decoder. Hence, cross-attention only occurs for encoder-decoder models. 

### Idea:

Encoder turns the input sequence into a set of rich representations (i.e. a set of token embeddings for each token — which will be used as a “memory”), while Decoder generates the output sequence (target) token-by-token, conditioned on the what it has generated so far (due to casual self-attention) and that Encoder’s “memory”.

Model at current token: “Given what I have generated so far, which parts of the source input matter right now?”

Cross-attention is exactly that lookup:

- Decoder states provide Queries (Q) —> “what do i need?”
    - Q represents the current need based on what is has generated so far  → “what am i looking for from the source to generate the next token given my current sequence / state ?”
- Encoder outputs provide Keys, Values (K, V) → “what source info is available”
    - K = “how should you match to me?”
    - V = “what information you get if you choose me”

## 7. Decoding Output

![image.png](image%2012.png)

The decoder’s outputs are fed into a final linear transformation layer and softmax function to transform them into a probability distribution over the word vector space. 

This probabiliy distribution represents the likelihood of each token being the next predicted token in the sequence. 

# Notes:

## Encoder, Decoder and Encoder-Decoder

Encoder-Decoder has cross-attention, but only inside the decoder block, while Encoder only and Decoder only models do not have cross-attention, but only has self-attention and casual self-attention respectively. 

Mental Model: 

Encoder-only = encoder-decoder minus decoder minus cross-attention

Decoder-only = encoder-decoder minus encoder minus cross-attention

## Inference

Models usually have L transformer blocks. Hence, the attention + FFN part runs L times for a single forward pass.

For next-token prediction:

- Sequence is passed through all layers once → get logits for the next token
- Sample / choose the next token
- Append it and run another forward pass

## Training

1. initialise random weights 
2. Take a batch of sequences
3. Forward pass through all blocks → logits for each position 
4. Compute loss (usually cross-entropy) 
5. Back-propagation to get gradients 
6. Update weights
7. Repeat till max epochs / validation stops improving. 

# References:

1. [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)
2. [https://huggingface.co/blog/Esmail-AGumaan/attention-is-all-you-need](https://huggingface.co/blog/Esmail-AGumaan/attention-is-all-you-need)
3. [https://aiml.com/explain-the-need-for-positional-encoding-in-transformer-models/](https://aiml.com/explain-the-need-for-positional-encoding-in-transformer-models/)
4. [https://stats.stackexchange.com/questions/565196/why-are-residual-connections-needed-in-transformer-architectures](https://stats.stackexchange.com/questions/565196/why-are-residual-connections-needed-in-transformer-architectures)
5. [https://www.baeldung.com/cs/transformer-networks-residual-connections](https://www.baeldung.com/cs/transformer-networks-residual-connections)
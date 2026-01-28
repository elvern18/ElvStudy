# LoRA (Low-Rank Adaption)

Technique for efficiently fine-tuning LLMs by introducing low-rank trainable weight matrices into specific model layers. 

# Problems with Traditional Fine-tuning

Fine-tuning is essentially the same as supervised training of a machine learning model — given a pre-trained model and labeled domain-specific dataset, we train the model on this specific dataset 

Fine-tuning speed depends on at least three factors:

1. Hardware Specification
2. Size of dataset for fine-tuning
3. Number of parameters in the model —> main issue addressed in LoRA

## Issues:

1. More parameters ⇒ more computation ⇒ more cost and time 
2. If fine-tune all parameters, might have unreliable predictions upon unseen data that deviates significant from the fine-tuning dataset
3. If have 2 unrelated domain-specific use cases, fine-tune 2 models, each for specific use case ⇒ have 2 fine-tuned models ⇒ inefficient memory management and storage

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*SmHiDIMW10lKXive.png)

# Adapters and Prefix Tuning (pre-LoRA)

## Adapters

- Collection of small, trainable parameters inside the pretrained model

### Before fine-tuning:

Freeze the weights of the pretrained model, but add adapters. 

### During fine-tuning:

Update only adapters inside

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*Pd8FJjqMc2zKUJZK.png)

Adapters can be easily removed and inserted back into the pretrained mode ⇒ if have multiple domain-specific use cases, only need to store one pretrained model + corresponding adapters instead of multiple fine-tuned models 

### Architecture

Adapter has 2 normal dense layers with activation function in between

- 1st dense layer downscales input
- 2nd dense layer upscales the output

Adapters are placed after each self-attention and FFN. 

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*7tde6IwERPspARJi.png)

### Problems with Adapters

There is no easy way to bypass the extra computation step of adapters. This means that the model with adapters can only be processed sequentially, resulting in increased latency during inference. 

## Prefix Tuning

Prefixes are another small set of parameters that are learnable embeddings, often initialised randomly or based on task-specific information. They act like additional prompts or context that can guide the model to perform better on new tasks. 

Same idea as adapters — freeze the weights of pretrained model and only update the embeddings of prefixes so that they can learn to generate task-relevant embeddings that condition the output of each layer 

![image.png](images/image.png)

*Figure: Normal fine-tuning (top) vs prefix fine-tuning (bottom).*

### Drawbacks:

1. Difficult to optimise 
2. Need to reserve some chunk of model’s sequence length for these prefixes in advance ⇒ reduces sequence length available to process a downstream task ⇒ negatively affect overall performance of the model after fine-tuning. 

# LoRA

## Hypothesis

The change-in-weight matrix $\Delta W$ has a low intrinsic rank ⇒ matrix can be represented using fewer dimensions using $A$ and $B$ matrices usch that $dim(BA) = dim(\Delta W) = dim(W)$

### Recall:

- Rank of matrix is the number of linearly independent rows / columns in that matrix
- Rank deficient matrices has linear dependencies and thus redundancy ⇒ can be represented more compactly

## Idea

Freeze the old weight matrix of the base model.

Take the change-in-weight matrix $\Delta W$ and approximate it with the product of 2 lower-rank matrices, $B$ and $A$, where $A$ is randomly initialised while $B$ is initialised to $0s$ such that $\Delta W = BA \approx 0$ at the start. $A$ and $B$ values are learnt during fine-tuning. 

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*0VXQdBY4fT85Ur9mwOZL2g.png)

### Formula

With hypothesis that change-in-weight matrix $\Delta W$ is low intrinsic rank $r$,

$B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k},\quad r \ll \min(d,k),\quad W \in \mathbb{R}^{d \times k}, \quad dim(BA) = dim(W)$

![image.png](images/image%201.png)

Note that $\Delta W$ gets added back to $W_0$

$$
h = W_0X + \triangle WX = W_0X + (BA)X
$$

![image.png](images/image%202.png)

BA as an approximation of the change-in-weight matrix (Wᵩ).

![image.png](images/image%203.png)

Comparison of rank-4 and rank-1 adaptation. LoRA results in fewer trainable parameters.

## Hyperparameters

Some important hyperparameters below:

### 1. Rank $r$

Low $r$ ⇒ cheaper, may underfit. High $r$ ⇒ more expensive but better fit

### 2. Scaling Alpha $\alpha$

When training, change-in-weight matrix is actually scaled by a factor of $\frac \alpha r$

$\Delta W = \frac \alpha r BA$

### 3. LoRA dropout

Dropout applied inside LoRA branch during training 

### 4. Target modules (where to apply LoRA)

Target weight matrices to apply LoRA. 

Common choices include:

- Attention projections (e.g. $Q, K, V$ matrices) — great for instruction / style steering
- MLP layers — more capacity for domain konwledge and reasoning shifts

## Pros

1. Cheaper to finetune as a tiny set of parameters is trained instead of the whole model ⇒ less GPU memory + faster training
2. Swappable — as weight updates are stored separately and we can store different adapters for different tasks, we can swap them accordingly without having multiple full models 
3. No inference latency as $W_{new} = W_0 + BA$ is the only extra step required i.e. simple dot product of small matrix & coordinate-wise sum to get the tuned weights $W_{new}$
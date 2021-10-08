# Style Transformer

## 1.What is the problem
1. Separating content from style (disentangling doesn’t guarantee the pure separation of content and style. Also, a good decoder can just overwrite the style into the input sentence)
2. Long-term dependency in RNNs -> results in loss of content (semantics)
    1. Fixed-size latent vector to encode input sentence (for disentanglement)
3. And of course most importantly, lack of parallel data

## 2.What is the novelty of the article
Idea of using entangled data rather than separating content & style was first proposed in another work (Multiple-Attribute Text Rewriting) but they tackled it with back-translation.
In this work: 
1. Use of transformers (specifically attention mechanism that doesn’t need fixed-size input sentence)
2. Use of discriminator network to learn input sentence as parallel data isn’t available 

## 3.How is this solved?
1. By using two transformer networks
    1. Style transformer network: a mapping function that outputs target sentence with the intended style. Input sentence and style are fed into network to reconstruct the input
    2. Discriminator network: learns to distinguish the style of different sentences and help style transformer network to control generated sentence’s style

## 4.The advantages and disadvantages of the experiment
Advantages:
1. Better performance in metrics such as content preserving, style transfer, and fluency
Disadvantages:
1. Less interpretability (no more latent representation) and maybe less control as well?!
2. Mostly about changing some attributes (or just single attribute) rather than rewriting the whole sentence => Authors suggest doing sth similar to “Multiple-attribute text rewriting” + using their back-translation method in the training process)

## 5.The problem that could be addressed in the future
Rewriting the whole sentence as explained in section 4.

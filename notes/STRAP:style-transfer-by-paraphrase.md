# Style Transformer

## 1. What is the problem?
Modern NLP defines the task of style transfer as modifying the style of a given sentence without appreciably changing its semantics, which implies that the outputs of style transfer systems should be paraphrases of their inputs. 
 However, many existing systems purportedly designed for style transfer inherently **warp the input’s meaning through attribute transfer, which changes semantic properties such as sentiment**.

## 2. What is the novelty of the article?
In this paper, we reformulate unsupervised style transfer as a **paraphrase generation problem**.

## 3. How is this solved?
**Fine-tuning pre-trained language models (GPT-2) on automatically generated paraphrase data** (So, no reinforcement learning, variational inference ,or autoregressive sampling during training). 
We fine-tune the large-scale pre-trained GPT2-large language model to implement both the paraphraser f_para and inverse paraphrasers for each style. Starting from a pre-trained LM improves both output fluency and generalization to small style-specific datasets. We use the encoder-free seq2seq modeling approach, where input and output sequences are concatenated together with a separator token. We use Hugging Face’s Transformers library to implement our models.

We create **pseudo-parallel sentence pairs** using a paraphrase model (Check Section 2.1) trained to maximize output diversity, 2) this paraphrasing step normalizes the input sentence by stripping away information that is predictive of its original style, 3) train an inverse paraphrase model specific to the original style, which attempts to generate the original sentence given its normalized version, 4) the model learns to identify and produce salient features of the original style without unduly warping the input semantics.


1. Create pseudo-parallel data by feeding sentences from different styles through a diverse paraphrase model (z = f_para(x) where x ∈ Xi) -> Now we have (X_i, Z_i) which is the pseudo-parallel dataset
2. Train style-specific inverse paraphrase models that convert these paraphrased sentences back into the original stylized sentences. Since f_para removes style identifiers from its input, the intuition behind this inverse paraphrase model is that it learns to **insert stylistic features through the reconstruction process**. Formally, the inverse paraphrase model f_{inv}^i for style i learns to reconstruct the original corpus X^i using the standard language modeling objective with cross-entropy loss LCE (Check section 2.2)
3. Use the inverse paraphraser for a desired style to perform style transfer

## 4. The advantages and disadvantages of the experiment.
**Advantages:**
Despite its simplicity, our method significantly outperforms state-of-the-art style transfer systems on both human and automatic evaluations.
This method is much more well-suited for the formality transfer since it doesn't just change a copule of attributes.

**Disadvantages:**
Need of a pre-trained model to be fine-tuned for the paraphrase generation, which by itself results in need for more data?

## 5. The problem that could be addressed in the future.
...
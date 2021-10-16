# S4L: Self-Supervised Semi-Supervised Learning
This work tackles the problem of semi-supervised learning of image classifiers. Our main insight is that the field of semi-supervised learning can benefit from the quickly advancing field of self-supervised visual representation learning. Unifying these two approaches, we propose the framework of self-supervised semi-supervised learning (S4L) and use it to derive two novel semi-supervised image classification methods. We demonstrate the effectiveness of these methods in comparison to both carefully tuned baselines, and existing semi-supervised learning methods. We then show that S4L and existing semi-supervised methods can be jointly trained, yielding a new state-of-the-art result on semi-supervised ILSVRC-2012 with 10% of labels.

## 1.What is the problem
Modern computer vision systems demonstrate outstanding performance on a variety of challenging computer vision benchmarks. Their success relies on the availability of a **large amount of annotated data** that is time-consuming and expensive to acquire. Moreover, applicability of such systems is typically limited in **scope defined by the dataset they were trained on**.

Self-supervised learning techniques define pretext tasks which can be formulated using only unlabeled data, but do require higher-level semantic understanding in order to be solved. As a result, models trained for solving these pretext tasks learn representations that can be used for solving other downstream tasks of interest, such as image recognition.

Despite demonstrating encouraging results, purely self-supervised techniques learn visual representations that are significantly inferior to those delivered by fully-supervised techniques. Thus, their practical applicability is limited and as of yet, self-supervision alone is **insufficient**.

## 2.What is the novelty of the article.
We hypothesize that self-supervised learning techniques could dramatically benefit from a small amount of labeled examples. By investigating various ways of doing so, we bridge self-supervised and semi-supervised learning, and propose a framework of semi-supervised losses arising from self-supervised learning targets.

+ proposing a new family of techniques for semi-supervised learning with natural images that leverage recent advances in self-supervised representation learning.
+ demonstrate that the proposed self-supervised semi-supervised (S4L) techniques outperform carefully tuned baselines that are trained with no unlabeled data, and achieve performance competitive with previously proposed semi-supervised learning techniques.
+ We further demonstrate that by combining our best S4L methods with existing semi-supervised techniques, we achieve new state-of-the-art performance on the semi-supervised ILSVRC-2012 benchmark.

## 3.How is this solved?
We focus on the semi-supervised image classification problem. Formally, we assume an (unknown) data generating joint distribution `p(X,Y)` over images and labels. The learning algorithm has access to a labeled training set `D_l` , which is sampled i.i.d. from `p(X,Y)` and an unlabeled training set `D_u` , which is sampled i.i.d. from the marginal distribution `p(X)`.
![Screenshot from 2021-10-16 13-20-25](https://user-images.githubusercontent.com/43045767/137583067-1a0e522c-34d4-4ce2-a273-afad4bebd225.png)

where `L_l` is a standard cross-entropy classification loss of all labeled images in the dataset, `L_u` is a loss defined on unsupervised images (we discuss its particular instances laterin this section), `w` is a non-negative scalar weight and `θ` is the parameters for model `f_θ(·)`. Note that the learning objective can be extended to multiple unsupervised losses.


### 3.1. Semi-supervised Baselines
In the following section, we compare S4L to several leading semi-supervised learning algorithms that are not based on self-supervised objectives. We now describe the approaches that we compare to.

Our proposed objective (Eq. 1) is applicable for semi supervised learning methods as well, where the loss `L_u` is the standard semi supervised loss as described below.
#### 3.1.1 Virtual Adversarial Training (VAT)
**The idea is making the predicted labels robust around input data point against local perturbation.** Concretely, the VAT loss for a model `f_θ` is:

![Screenshot from 2021-10-16 13-29-06](https://user-images.githubusercontent.com/43045767/137583265-cf5ac651-5d0d-423f-a9cf-f0b5bb9b6d28.png)

#### 3.1.2 Conditional Entropy Minimization (EntMin)
This works under the assumption that **unlabeled data indeed has one of the classes that we are training on**, even when the particular class is not known during training. It adds a loss for unlabeled data that, when minimized, encourages the model to make confident predictions on unlabeled data. Specifically, the conditional entropy minimization loss for a model `f_θ` (treating `f_θ` as a conditional distribution of labels overimages) is

![Screenshot from 2021-10-16 13-36-50](https://user-images.githubusercontent.com/43045767/137583452-a926cd20-235a-4719-b2f4-31a1c87dc518.png)

Alone, the _EntMin_ loss is not useful in the context of deep neural networks because the model can easily become extremely confident by increasing the weights of the last layer. One way to resolve this is to encourage the model predictions to be **locally-Lipschitz** (i.e has bounded derivative), which _VAT_ does. Therefore, we only consider _VAT_ and EntMin combined, not just _EntMin_ alone.

#### 3.1.3 Pseudo-Label
It is a simple approach: Train a model only on labeled data, then make predictions on unlabeled data. Then enlarge your training set with the predicted classes of the unlabeled data points whose predictions are confident past some threshold of confidence. Retrain your model with this enlarged labeled dataset. While shows that in a simple ”two moons” dataset, psuedo-label fails to learn a good model, in many real datasets this approach does show meaningful gains.

## 4.The advantages and disadvantages of the experiment
As you can see S4L accuraccy is outperforming self-supervised methods as showing in bellow figure:

![Screenshot from 2021-10-16 12-45-54](https://user-images.githubusercontent.com/43045767/137582409-5ffffdac-1d48-4c59-aa1f-bf5581feac1c.png)

## 5.The problem that could be addressed in the future.
We instantiated two such methods: S4L-Rotation and S4L-Exemplar and have shown that they perform **competitively** to methods from the semi-supervised literature on the challenging ILSVRC-2012 dataset. We further showed that S4L methods are complementary to existing semi-supervision techniques, and MOAM, our proposed combination of those, leads to state-of-the-art performance.

While all of the methods we investigated show promising results for learning with 10 % of the labels on ILSVRC2012, the picture is much less clear when using only 1 %. It is possible that in this low data regime, when only 13 labeled examples per class are available, the setting fades into the few-shot scenario, and a very different set of methods would be required for reaching much better performance.

Nevertheless, we hope that this work inspires other researchers in the field of self-supervision to consider extending their methods into semi-supervised methods using our S4L framework, as well as researchers in the field of semi-supervised learning to take inspiration from the vast amount of recently proposed self-supervision methods.

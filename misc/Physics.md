# Learning Physics

---

- [**Real time fluid simulation and control using the Navier-Stokes equations MSc thesis (2012) – Károly Zsolnai-Fehér – Research Scientist**](https://users.cg.tuwien.ac.at/zsolnai/gfx/fluid_control_msc_thesis/)
- [**[2002.09405] Learning to Simulate Complex Physics with Graph Networks**](https://arxiv.org/abs/2002.09405)
- [**Learning to simulate**](https://sites.google.com/view/learning-to-simulate/home#h.p_hjnaJ6k8y0wo)
- [**Physics-aware Difference Graph Networks for Sparsely-Observed Dynamics **](https://openreview.net/forum?id=r1gelyrtwH)
- [**Understanding and mitigating gradient pathologies in physics-informed neural networks**](https://arxiv.org/abs/2001.04536v1.pdf)
- [**DiffTaichi: Differentiable Programming for Physical Simulation**](https://arxiv.org/abs/1910.00935.pdf)
- [**PyTorch implementation of Approximate Derenderer, Extended Physics, and Tracking (ADEPT)**](https://github.com/JerryLingjieMei/ADEPT-Model-Release)
- [**Graph Networks as Learnable Physics Engines for Inference and Control**](https://arxiv.org/abs/1806.01242.pdf)

- [**A Compositional Object-Based Approach to Learning Physical Dynamics**](https://arxiv.org/abs/1612.00341.pdf)
- [**Interaction Networks for Learning about Objects, Relations and Physics**](http://papers.nips.cc/paper/6418-interaction-networks-for-learning-about-objects-relations-and-physics.pdf)
- [**Learning to See Physics via Visual De-animation**](http://papers.nips.cc/paper/6620-learning-to-see-physics-via-visual-de-animation.pdf)
- [**Graph Networks as Learnable Physics Engines for Inference and Control**](https://arxiv.org/abs/1806.01242.pdf)
- [**Flexible neural representation for physics prediction**](http://papers.nips.cc/paper/8096-flexible-neural-representation-for-physics-prediction.pdf)
- [**Learning to Decompose and Disentangle Representations for Video Prediction**](http://papers.nips.cc/paper/7333-learning-to-decompose-and-disentangle-representations-for-video-prediction.pdf)
- [**Relational inductive bias for physical construction in humans and machines**](https://arxiv.org/abs/1806.01203.pdf)
- [**A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning**](http://papers.nips.cc/paper/6951-a-disentangled-recognition-and-nonlinear-dynamics-model-for-unsupervised-learning.pdf)
- [**visual-interaction-networks-learning-a-physics-simulator-from-video.pdf**](http://papers.nips.cc/paper/7040-visual-interaction-networks-learning-a-physics-simulator-from-video.pdf)
- [**Neural Relational Inference for Interacting Systems**](https://arxiv.org/abs/1802.04687.pdf)
- [**Chang: A compositional object-based approach to learning... - Google Scholar**](https://scholar.google.co.in/scholar?start=0&hl=en&as_sdt=2005&sciodt=0,5&cites=9706972547667418204&scipsc=)
- A Compositional Object-Based Approach to Learning Physical Dynamics https://arxiv.org/abs/1612.00341.pdf
- Learning by Abstraction: The Neural State Machine https://arxiv.org/abs/1907.03950.pdf
- 7040-visual-interaction-networks-learning-a-physics-simulator-from-video.pdf

---

The main objective for the creation of our benchmark (CoPhy: Counterfactual Learning of Physical Dynamics) is :

(a) to focus specifically on evaluating capabilities of state of the art models for performing counterfactual reasoning,
(b) to be unbiased in terms of distributions of parameters to be estimated and balanced with respect to possible outcomes, and (c) to have sufficient variety in terms of scenarios and latent physical characteristics of the scene that are not visually observed and therefore can act
as confounders.
To the best of our knowledge, none of existing intuitive physics benchmarks have these properties.

IntPhys (Riochet et al., 2018) focuses on a high level task of estimating physical plausibility in a black box fashion and modeling out of distribution events at test time. CATER(Girdhar & Ramanan, 2019) introduces a video dataset requiring spatiotemporal understanding in order to solve the tasks such as action recognition, compositional action recognition, and adversarial target tracking.

Phyre (Bakhtin et al., 2019) is an environment for solving physics based puzzles, where achieving sample efficiency may implicitly require counterfactual reasoning, but this component is not explicitly evaluated, construction of parallel data with several alternative outcomes is
not straightforward, and the trivial baseline performance levels are not easy to estimate. Adapting these benchmarks to counterfactual reasoning would require significant refactoring and changing the logic of the data sampling.

CLEVRER (Yi et al., 2019) is a diagnostic video dataset for systematic evaluation of models on a wide range of reasoning tasks including counterfactual questions. They cast the task as a classification problem where the model has to choose between a set of possible answers, whereas our benchmark requires the predicting the dynamics of each object.

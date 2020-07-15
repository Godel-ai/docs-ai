## Reinforcement Learning

- Model Predictive Control: Models a PID controller? where Integral is the memory of the previous errors (keep track of past mistakes), Differential predicts the trajectory of the current work.

---
- World Models
- Embodied AI 
- Sim2Real
  
Modular control strategies, multi-task control : using control as a reward

what is the meaning of reinforcement learning → on Gecco creatures → does it transfer to the real world → what does it mean control such a creature in the fake world → if we build its real world counterpart, will it work as good?

How good actually is dexterous hand manipulation → how much is that transferable automatically to other tasks ?

How good is hierarchical goal-setting → generating new goals → to complete tasks → does it generalise?

What is the state of multi-task learning today → is it better than control theory ?

Policy Gradients is not data-efficient and Model-based Reinforcement Learning does not work that well ?

What is the point of training such a humongous network as the one of Dota-Five (only the collaboration part may be important) by wasting so much compute and using the same algorithm (PPO) if it does not transfer?

What purpose does playing these well-controlled environment games may be useful in the real world? A war or an entertainment source?

Can we integrate the intuition and understanding of the world from computer vision and NLP (knowledge base) to this reinforcement world to make it more efficient?

How do we add constraints to the actions the agent can take ?

---

## Self-Supervised Exploration via Disagreement ##
#### [[Project Website]](https://pathak22.github.io/exploration-by-disagreement/) [[Demo Video]](https://youtu.be/POlrWt32_ec)

---

## Zero-Shot Visual Imitation ##
#### In ICLR 2018 [[Project Website]](https://pathak22.github.io/zeroshot-imitation/) [[Videos]](http://pathak22.github.io/zeroshot-imitation/index.html#demoVideos)

We propose an alternative paradigm wherein an agent first explores the world without any expert supervision and then distills its experience into a goal-conditioned skill policy with a novel forward consistency loss. The key insight is the intuition that, for most tasks, reaching the goal is more important than how it is reached.

---

## World Models Experiments

Step by step instructions of reproducing [World Models](https://worldmodels.github.io/) ([pdf](https://arxiv.org/abs/1803.10122)).

![World Models](https://worldmodels.github.io/assets/world_models_card_both.png)

Please see [blog post](http://blog.otoro.net//2018/06/09/world-models-experiments/) for step-by-step instructions.

---
# Stream AC algorithm comparison to Classic AC for various entropy coefficient values in CartPole-v1 environment


Implementation based on the <em>[Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)</em> paper by [Mohamed Elsayed](http://mohmdelsayed.github.io), [Gautham Vasan](https://gauthamvasan.github.io), and [A. Rupam Mahmood](https://armahmood.github.io). 

In streaming reinforcement learning, an agent receives an observation and reward at each step, takes an action, and makes a learning update immediately without storing the sample.  This approach is ideal for resource-constrained, real-time applications but poses significant challenges, as deep RL algorithms typically rely on replay buffers to store and reuse past experiences. Existing deep RL methods often experience learning instabilities and failures in streaming settings, a phenomenon known as the stream barrier.
The Stream – X algorithms address these issues by introducing five key techniques:
-Sparse initialization to reduce interference between dissimilar inputs.
-Eligibility traces for better credit assignment.
-Overshooting-bounded Gradient Descent (ObGD) to prevent large, destabilizing updates.
-Layer Norm to stabilize activation distributions.
-Data scaling to normalize observations and rewards.
Entropy regularization is a technique that promotes exploration by penalizing low entropy policies. In this experiment, I evaluate how different entropy coefficient values (τ) affect the Stream AC algorithm performance in comparison to the Classic AC in the CartPole-v1 environment.

## Results
<p float = "center">
<img src="https://github.com/user-attachments/assets/cfe7f8d8-de57-4fde-b569-4b67cc39a1d9" width="300" />
<img src="https://github.com/user-attachments/assets/f213798e-e46b-478f-85ab-2ce056a40f79" width="300" />
</p>

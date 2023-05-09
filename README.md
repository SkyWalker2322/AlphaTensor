# Discovering-faster-matrix-multiplication-algorithms-with-RL
BTP 2023

### Introduction:
AlphaTensor is a system developed by DeepMind to automatically discover new, more efficient matrix multiplication algorithms using deep reinforcement learning. Matrix multiplication is essential to many fields, including computer graphics, digital communications, scientific computation, and neural networks. AlphaTensor can search for efficient algorithms tailored to specific hardware without prior knowledge and is capable of autonomously discovering provably correct matrix multiplication algorithms. It employs TensorGame, a single-player game that is played using techniques developed to self-learn board games such as Chess or Go. The player manipulates a given input tensor to create a set of instructions representing a new multiplication algorithm. Tensors can represent any bilinear operation, and extensions of AlphaTensor targeting other mathematical problems could unlock new possibilities for research in complexity theory and other areas of mathematics. AlphaTensor operates by searching the large combinatorial space of potential algorithms using reinforcement learning techniques.

The concept of tensors, which are a generalization of matrices, is very crucial over here. Tensors are higher-dimensional objects that can be thought of as a collection of numbers arranged in a box-like structure, with each number indexed by multiple indices. Matrices can be seen as a special case of tensors, specifically 2D tensors.

The paper goes on to explain how matrix multiplication can be represented by a 3D tensor, and how finding new matrix multiplication algorithms is equivalent to finding decompositions of this tensor. The article also mentions a large number of possible decompositions, and how sophisticated strategies are needed to explore this space. The AlphaTensor approach is introduced as a 3-dimensional board game called TensorGame.

### Tensor Game
TensorGame is a single-player 3D board game that models tensor decomposition as a reinforcement learning problem. The goal of the game is to find a decomposition of a given tensor T as a sum of R outer products with R as small as possible. The game starts with the initial state set to be the target tensor T. In each step, the player selects three vectors u(t), v(t), and w(t), and the state is updated recursively. 
<img src="/resources/image5.jpeg" alt="Alt text" display="centre"><br>
The game ends after R steps, and the player wins if the resulting tensor is zero.
<img src="/resources/image4.jpeg" alt="Alt text">
Negative rewards are applied at each step, encouraging the player to reach the zero-tensor in fewer steps. An additional negative reward is applied if the player terminates with a non-zero vector after a preset maximum number of moves. The game provides a framework to explore the combinatorial space of tensor decomposition in a systematic way.
### MCTS
Monte Carlo Tree Search is a tree search algorithm invented in 2007, generally used in more complex games and huge action spaces which is ideal for our project. MCTS algorithm consists of 4 steps:
- Selection: Selecting good child nodes, starting from the root node R, that represent states leading to better overall outcomes (win).
- Expansion: If L is not a terminal node (i.e. it does not end the game), then create one or more child nodes and select one (C).
- Simulation(Rollout): Run a simulated playout from C until a result is achieved.
- Backpropagation: Update the current move sequence with the simulation result.
#### Upper Confidence Bound (UCB)
Selection: Similar to the Best first search, MCTS uses the UCB value of each node which gives the notion of Confidence in each respective node to attain a solution(win).

<img src="/resources/image3.jpeg" alt="Alt text">

Where, Q(s, a) is the Action-Value function, 𝞹(s, a) is the empirical Policity probability distribution function, N(s, a) is the Visit count of the child Node, c(s) is the exploration factor that influences the Exploration Vs Exploitation.

Simulation: whenever the leaf node is reached for the first time, a rollout will be made where the Value of the current state is computed using the Neural Network. 

Backpropagation: The value of the current state is backpropagated to the root node and the visit count, Q values of each node are updated.

### Neural Network
A transformer-based neural network architecture is used. The input is the history of episodes generated till that node.
<img src="/resources/image2.jpeg" alt="Alt text">
The neural network is composed of a common torso (acting as an encoder), followed by a double head. More specifically:
<ol>
<li><b>Torso:</b> The torso is composed of a stack of 3 Transformer Encoder layers. Each layer is composed of a Multi-Head Attention layer followed by a Feed Forward layer. The torso is based on a modification of transformers that utilizes a special form of attention mechanism called axial attention which is a self-attention-based autoregressive model like an encoder. Its purpose is to create a representational embedding of the input (as shown in the above figure) that is useful for both policy and value heads.
<img src="/resources/image1.jpeg" alt="Alt text">

<li><b>Policy Head: </b>  The policy head's purpose is to predict a distribution of potential actions. It uses a transformers architecture to model an autoregressive policy. Autoregressive here means that the model acts by measuring the correlation between observations at previous time steps to predict the output (similar to a decoder architecture in language models).

<li><b>Value Head: </b> The value head is composed of a multilayer perceptron which is a fully connected class of feedforward artificial neural network (ANN) trained to predict a distribution of the returns from the current state (cumulative reward).
</ol>

### Code
The working code for discovery of matrix multiplication algorithms can be found in ```./Algos/src/``` directory. Make sure you have all the requirements mentioned in the README file
import Coach
import Environment
import NeuralNet
import torch

def main():

    env = Environment.Environment()
    
    nnet = NeuralNet.NeuralNet(
        input_size=env.get_init_state().tensor.shape,
        num_actions=env.num_actions,
        lr=1e-3
    )
    # nnet.torso.load_state_dict(torch.load("./parameters/torso_parameters.pth"))
    # nnet.policy_head.load_state_dict(torch.load("./parameters/policy_head_parameters.pth"))
    # nnet.value_head.load_state_dict(torch.load("./parameters/value_head_parameters.pth"))
    alphazero = Coach.Coach(
        env,
        nnet,
        num_iter=30,  # number of alternations between self-playing and nnet training
        num_self_play_games=100,  # number of self-play games per iteration
        num_mcts_simulations=100,  # number of simulations during "thinking over the move"
        max_num_steps=8,  # per game
        max_num_examples=1000,  # max length of history of moves, which network is training on
        num_epochs=10,  # number of epochs of nnet training per iteration
        batch_size=50,
        logs_dir='./logs_2/',
        device=0
    )
    alphazero.learn()
    torch.save(nnet.torso.state_dict(),"./parameters/torso_parameters.pth")
    torch.save(nnet.policy_head.state_dict(),"./parameters/policy_head_parameters.pth")
    torch.save(nnet.value_head.state_dict(),"./parameters/value_head_parameters.pth")


if __name__ == '__main__':
    main()

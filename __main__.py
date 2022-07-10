import agent_q_learning
from bokeh.plotting import figure, show
import numpy as np


def training_sequence(agent):

    agent.alpha_init = .5
    num_repeats = 100
    agent.k_alpha = 5e4
    agent.k_epsilon = 1000
    agent.gamma = .96
    agent.max_training_visits = 1e5
    agent.epsilon_init = .2
    training_loss = agent.train_on_examples(num_repeats)

    p = figure(
               title=(
                      f'Training Loss: init_alpha={agent.alpha_init},'
                      + f'k_alpha={agent.k_alpha}, gamma={agent.gamma}'
                      ),
               x_axis_label='Training Runs',
               y_axis_label='Loss',
               ) 
    p.line(range(len(training_loss)), training_loss)
    show(p)

    agent.alpha_init = .25
    agent.epsilon_init = 0
    agent.max_episodes = 10000
    agent.train()

    convergence = np.array(agent.convergence)

    p2 = figure(
                title='Convergence Plot',
                x_axis_label='Training Runs',
                y_axis_label='Agent Score',
                )
    p2.line(convergence[:, 0], convergence[:, 1])
    show(p2)

    #agent.save_training()


if __name__ == '__main__':

    agent = agent_q_learning.Agent_Q_Learning()
    training_sequence(agent)
    #agent.record_training_example()


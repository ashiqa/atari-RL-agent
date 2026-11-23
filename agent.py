# External libraries
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict

# Local modules
from environment import PongEnviroment
from utils import parse_args, set_seed, plot_train_rewards, plot_evaluation_rewards

class Policy(nn.Module):
    def __init__(self, state_space: int, action_space: int, hidden_dim: int):
        super().__init__()
        # Define policy network architecture
        # Read the pytorch documentation on nn.Sequential to learn how to use it
        self.network = nn.Sequential(OrderedDict([

            # Fill with the necessary layers
            ('policy_layer_1', nn.Linear(state_space, hidden_dim)),
            ('policy_activation_1', nn.ReLU()),
            ('policy_layer_2', nn.Linear(hidden_dim, hidden_dim)),
            ('policy_activation_2', nn.ReLU()),
            ('policy_layer_3', nn.Linear(hidden_dim, action_space)),
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        # Implement forward pass
        return self.network(state)

class Value(nn.Module):
    def __init__(self, state_space: int, hidden_dim: int):
        super().__init__()
        # Define policy network architecture
        # Read the pytorch documentation on nn.Sequential to learn how to use it
        self.network = nn.Sequential(OrderedDict([

            # Fill with the necessary layers
            ('value_layer_1', nn.Linear(state_space, hidden_dim)),
            ('value_activation_1', nn.ReLU()),
            ('value_layer_2', nn.Linear(hidden_dim, hidden_dim)),
            ('value_activation_2', nn.ReLU()),
            ('value_layer_3', nn.Linear(hidden_dim, 1)),
        ]))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Implement forward pass
        return self.network(state)


class Agent(nn.Module):
    def __init__(
        self,
        state_space: int,
        action_space: int,
        gamma: float,
        lr: float,
        max_training_steps: int,
    ):
        """
        Initialize the agent.

        Parameters:
            state_space      (tuple) : Size of the state space
            action_space       (int) : Size of the action space
            gamma            (float) : Discount factor
            lr               (float) : Learning rate
            max_training_steps (int) : Max steps in training episodes
        """
        super(Agent, self).__init__()

        # Initialize parameters
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.lr = lr
        self.max_training_steps = max_training_steps

        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # Initialize policy network (actor)
        # Refer to the handout for the value of hidden_dim
        self.policy = Policy(state_space, action_space, 256)
        
        # Initialize value network (critic)
        # Refer to the handout for the value of hidden_dim
        self.value = Value(state_space, 256)
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr = self.lr)
        
        # Precalculate cummulative discount factors
        self.cum_gamma = self.gamma ** torch.arange(10)

    
    def get_action(self, state: torch.Tensor) -> int:
        """
        Helper function to get the action the agent will take given the state
        of the environment.

        IMPORTANT: Should only be used when deploying the agent.

        Parameters:
            state (torch.Tensor) : State encoded as tensor.

        Returns:
            Returns an integer denoting the action the selected action.
        """
        # HINT: Remember how to stop gradient propagation
        with torch.no_grad():
            # Sample action from the policy
            logits = self.policy.forward(state)
            prob = torch.nn.functional.softmax(logits, dim=-1)
            return torch.multinomial(prob, 1).item()
        
  

    def n_step_returns(self, n: int, rewards: torch.Tensor, next_state_values: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        """
        Calculates the N-Step-Returns for every timestep

        n_step_returns_t = r_t + r_t+1 * gamma + r_t+2 * gamma ^ 2 + ... + r_t+n-1 * gamma ^ n-1 + V(s_t+n) * gamma ^ n

        Parameters:
            rewards (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_state_values (torch.Tensor): Estimated value for states at timesteps t+1, t+2, ...
            terminated (torch.Tensor): True for the terminal states in next_states_values
        """
        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # TODO: Calculate the N-Step-Returns
        t = rewards.shape[1]
        n = min(n, t)
        gamma_vec = (self.gamma ** torch.arange(t)).reshape(1, -1, 1)
        discounted_rewards = rewards * gamma_vec
        padded_discounted_rewards = torch.nn.functional.pad(discounted_rewards, pad=(0, 0, 1, n - 1))
        cumm_rewards = torch.Tensor.cumsum(padded_discounted_rewards, dim = 1)
        window_sub = cumm_rewards[:, n:, :] - cumm_rewards[:, :-n, :]
        reward_sum = window_sub / gamma_vec

        bootstrap_idx = torch.clamp(torch.arange(t) + n - 1, max=t - 1)
        shifted_val = next_state_values[:, bootstrap_idx, :]
        shifted_term = terminated[:, bootstrap_idx, :]
        shifted_val = shifted_val * (~shifted_term)
        bootstrap = (self.gamma ** n) * shifted_val

        # Hint: Use torch.Tensor.cumsum, torch.nn.functional.pad and self.cum_gamma for efficient implementation
        # IMPORTANT: Read the pytorch documentation for these functions carefully, especially for torch.nn.functional.pad
        return reward_sum + bootstrap

    def value_loss(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss used to optimize the value network (critic)

        Parameters:
            states        (torch.Tensor): States at timesteps t, t+1, ...
            rewards       (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_states   (torch.Tensor): States at timesteps t+1, t+2, ...
            terminated    (torch.Tensor): True for the terminal states in next_states
        """
        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # TODO: Calculate the loss for the value network (critic)
        # Hint: Remember how to stop gradient propagation for the target calculation
        # Refer to the handout for the value of n
        with torch.no_grad():
            next_state_values = self.value(next_states)
            returns = self.n_step_returns(10, rewards, next_state_values, terminated)
        values = self.value(states)
        return torch.nn.functional.mse_loss(values, returns)

    def update_value(self,) -> None:
        """
        Updates the parameters of the value network (critic) by performing
        a step of gradient descent and sets all gradients to zero.
        """
        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # Use the value optimizer to update the parameters
        # Use the value optimizer to set all the gradients to zero
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()

    def policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss used to optimize the policy network (actor)

        Parameters:
            states        (torch.Tensor): States at timesteps t, t+1, ...
            actions       (torch.Tensor): States at timesteps t, t+1, ...
            rewards       (torch.Tensor): Rewards at timesteps t, t+1, ...
            next_states   (torch.Tensor): States at timesteps t+1, t+2, ...
            terminated    (torch.Tensor): True for the terminal states in next_states
        """
        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # TODO: Calculate the loss for the policy network (actor)
        # Hint: Remember how to stop gradient propagation for the target calculation
        # Refer to the handout for the value of n
        with torch.no_grad():
            next_state_values = self.value(next_states)
            returns = self.n_step_returns(10, rewards, next_state_values, terminated)
            values = self.value(states)
            advantages = returns - values
        logits = self.policy(states)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions)
        return (-advantages * action_log_probs).mean()

    def update_policy(self) -> None:
        """
        Updates the parameters of the policy network (actor) by performing
        a step of gradient descent and sets all gradients to zero.
        """
        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # Use the policy optimizer to update the parameters
        # Use the policy optimizer to set all the gradients to zero
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

    def save(self, filename: str) -> None:
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load(self, filename: str) -> None:
        checkpoint = torch.load(filename, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])

def deploy_agent(
    agent: Agent, env: PongEnviroment,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the agent in the environment for one episode

    Parameters:
        agent    (Agent) : Reinforcement Learning Agent
        env (Environment): Environment

    Returns:
        state       (torch.FloatTensor) : States at each timestep with shape (N, T, S)
        action      (torch.IntTensor)   : Actions at each timestep with shape (N, T, 1)
        reward      (torch.FloatTensor) : Rewards after each action with shape (N, T, 1)
        new_state   (torch.FloatTensor) : New states after each action with shape (N, T, S)
        terminated  (torch.BoolTensor)  : Whether the new state is a terminal state (N, T, 1)
    """
    # ====== BEGIN STUDENT SOLUTION ===========================

    # ====== END STUDENT SOLUTION ===========================
    # Reset the environment 
    state = env.reset()

    states_list = []
    actions_list = []
    rewards_list = []
    new_states_list = []
    terminated_list = []

    # Start trajectory
    done = False
    while not done:
        # Get action from agent
        action = agent.get_action(state)

        # Take step in environment
        next_state, reward, terminated, truncated = env.step(action)

        # Store trajectory step
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        new_states_list.append(next_state)
        terminated_list.append(terminated)

        # Advance to the next state
        state = next_state

        # Stop the trajectory if the episode got truncated or terminated
        if truncated or terminated:
            done = True

    # IMPORTANT: Check the shapes and data types for the return values in the docstring  
    states = torch.stack(states_list).unsqueeze(0)
    new_states = torch.stack(new_states_list).unsqueeze(0)
    actions = torch.tensor(actions_list, dtype=torch.int64).reshape(1, -1, 1)
    rewards = torch.tensor(rewards_list, dtype=torch.float).reshape(1, -1, 1)
    terminated = torch.tensor(terminated_list, dtype=torch.bool).reshape(1, -1, 1)

    return states, actions, rewards, new_states, terminated


def main():
    args = parse_args()

    if args.eval_only == False:
        set_seed(10301) # DON'T DELETE THIS

        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # Initialize environment
        env = PongEnviroment(max_steps=args.max_steps)

        # Set random seed (DON'T DELETE THIS)
        env.reset(seed=10301)

        state_size = sum(env.observation_space.shape)
        action_size = int(env.action_space.n)

        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # Initialize agent
        agent = Agent(state_size, action_size, args.gamma, args.lr, args.max_steps)

        # Train the agent
        train_rewards_list = []
        eval_rewards_list = []
        for episode in tqdm(
            range(args.train_episodes), "Training episodes", leave=False
        ):
            # ============ Training ===========
            # ====== BEGIN STUDENT SOLUTION ===========================

            # ====== END STUDENT SOLUTION ===========================
            # Deploy the current policy to get a new trajectory
            states, actions, rewards, new_states, terminated = deploy_agent(agent, env)

            # Store training metrics
            train_mean_return = rewards.sum().item()
            train_rewards_list.append(train_mean_return)

            # ====== BEGIN STUDENT SOLUTION ===========================

            # ====== END STUDENT SOLUTION ===========================
            # Calculate the loss for the policy and value
            # Hint: Check the agent's methods
            value_loss = agent.value_loss(states, rewards, new_states, terminated)
            policy_loss = agent.policy_loss(states, actions, rewards, new_states, terminated)

            # Scale losses and get the gradients
            value_loss = value_loss / args.batch_size
            policy_loss = policy_loss / args.batch_size
            value_loss.backward()
            policy_loss.backward()

            if (episode + 1) % args.batch_size == 0:
                # ====== BEGIN STUDENT SOLUTION ===========================

                # ====== END STUDENT SOLUTION ===========================
                # Update the policy and value functions
                agent.update_policy()
                agent.update_value()

            # ============ Evaluation ===========
            if (episode + 1) % args.eval_every == 0:
                eval_mean_undiscounted_return = 0
                for eval_episode in range(args.eval_episodes):
                    # ====== BEGIN STUDENT SOLUTION ===========================

                    # ====== END STUDENT SOLUTION ===========================
                    # Deplay the agent
                    states, actions, rewards, new_states, terminated = deploy_agent(agent, env)

                    # Store evaluation metrics
                    T = rewards.shape[2]
                    eval_undiscounted_return = rewards.sum().item()
                    eval_mean_undiscounted_return += eval_undiscounted_return / args.eval_episodes
                eval_rewards_list.append(eval_mean_undiscounted_return)

            # Store network checkpoints
            if episode + 1 % args.store_every == 0:
                print("Storing checkpoint")
                agent.save("checkpoint.pth")

        # Store final network
        agent.save("checkpoint.pth")

        # Plot returns and moving average vs episodes
        train_rewards_array = np.array(train_rewards_list)
        eval_rewards_array = np.array(eval_rewards_list)
        plot_train_rewards(train_rewards_array)
        plot_evaluation_rewards(eval_rewards_array, args.eval_every)

    else:
        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # Initialize environment with record = True
        env = PongEnviroment(max_steps=args.max_steps, record=True)

        state_size = sum(env.observation_space.shape)
        action_size = int(env.action_space.n)

        # ====== BEGIN STUDENT SOLUTION ===========================

        # ====== END STUDENT SOLUTION ===========================
        # Initialize agent
        agent = Agent(state_size, action_size, args.gamma, args.lr, args.max_steps)

        # Load networks
        agent.load("checkpoint.pth")

        # Evaluate the agent
        N = 20
        trajectory_indices = torch.arange(0, N)
        rewards_list = []
        for i in range(N):
            # ====== BEGIN STUDENT SOLUTION ===========================

            # ====== END STUDENT SOLUTION ===========================
            # Deploy the agent to get a trajectory
            states, actions, rewards, new_states, terminated = deploy_agent(agent, env)
            rewards_list.append(rewards)
        env.close()

        # Get the total reward for each trajectory
        rewards = torch.tensor([rewards.sum() for rewards in rewards_list])
        # Get the length of each trajectory
        steps = torch.tensor([rewards.shape[1] for rewards in rewards_list])

        # Identify the longest winning trajectory
        sorted_steps, sort_indices = steps.sort()
        rewards = rewards[sort_indices]
        trajectory_indices = trajectory_indices[sort_indices]
        longest_winning_trajectory = trajectory_indices[rewards > 0][-1].item()
        print(f"Longest winning trajectory: {longest_winning_trajectory}")

if __name__ == "__main__":
    main()

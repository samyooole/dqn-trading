
from qNetwork import qNetwork, ExperienceReplayBuffer
from tradingEnvironments import purePriceEnvironment
import random
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

class dqnTrader():

    # simply: just one stock first, say AAPL

    def __init__(self, tradingEnvironment, training_interval, update_interval, γ, batch_size, learning_rate=0.001):

        self.tradingEnvironment = tradingEnvironment
        input_size = tradingEnvironment.states_dim
        output_size = tradingEnvironment.actions_dim
        self.actions = self.tradingEnvironment.actions
        """
        Initialize dqn tools
        """
        ## input/output comes from tradingEnvironment
        self.q_network = qNetwork(input_size, output_size)
        self.target_network = qNetwork(input_size, output_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.epoch = 0
        self.training_interval = training_interval
        self.update_interval = update_interval
        self.erBuffer = ExperienceReplayBuffer(batch_size)
        self.γ = γ
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.ε = 1.0  # initial exploration rate
        self.ε_min = 0.01  # minimum exploration rate
        self.ε_decay = 0.995 # decrease exploration rate as the agent becomes good at trading

    def step(self, state):
        """
        If step is a training step, call train()
        Else:
            With probability ε, select a random action
            With probability 1-ε, pick the action with the highest expected future reward according to Q(s,a)
        """

        training = (self.epoch % self.training_interval == 0) & (self.epoch !=0)

        if training:
            self.train()

        if not training:

            if random.random() < self.ε:
                # select a random action
                trade=  random.choice(self.actions) ############## HWUC: random.choice(self.tradingEnvironment.actions)
                action_index = self.actions.index(trade)
            else:
                # get action with largest value

                with torch.no_grad():
                    q_values = self.q_network(state)
                    Q, action_index = q_values.max(0)

                    trade = self.actions[action_index]
                
            # execute chosen action
            reward, next_state = self.tradingEnvironment.dynamics(state, trade, self.timestep) ####### HWUC: self.tradingEnvironment."dynamics" - should generically take in state and action, return reward and next state

            # check if terminal
            terminal = self.timestep == self.tradingEnvironment.total_timesteps - 2 ########### HWUC: establish some generic total_timesteps in tradingEnvironment

            # store in replay buffer
            add_to_buffer = {'state': state, 'action_index': action_index, 'reward': reward, 'next_state': next_state, 'terminal':terminal}

            self.erBuffer.add(add_to_buffer)
            self.timestep+=1
        self.epoch += 1

       
    
    
    def train(self):
        # Train the underlying Q-network
        # not (directly) dependent on system dynamics which is great!
        batch = self.erBuffer.sampleBuffer()

        # Initialize lists to store Q-values and target Q-values for all transitions in the batch
        q_values_batch = []
        target_q_values_batch = []

        for transition in batch:
            state, action_index, reward, next_state, terminal = transition['state'], transition['action_index'], transition['reward'], transition['next_state'], transition['terminal']

            # Compute Q-values for the current state
            q_values = self.q_network(state)

            # Append Q-value for the action taken to the list
            q_value = q_values[action_index]
            q_values_batch.append(q_value)

            # Compute target Q-value
            if terminal:
                target_q_value = reward
            else:
                with torch.no_grad():
                    q_prime_values = self.target_network(next_state)
                    max_q_prime, _ = q_prime_values.max(0)  # Max Q-value for the next state
                    target_q_value = reward + self.γ * max_q_prime

            # Append target Q-value to the list
            target_q_values_batch.append(target_q_value)


            # Do fitting every single time
            loss = nn.MSELoss()(q_value, target_q_value)
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("Loss: ", str(loss.item()))

        """
        # Convert lists to tensors
        q_values_batch = torch.stack(q_values_batch)
        target_q_values_batch = torch.stack(target_q_values_batch)

        # Compute loss (MSE between predicted Q-values and target Q-values for the entire batch)
        loss = nn.MSELoss()(q_values_batch, target_q_values_batch)

        # Print the loss
        print("Loss: ", str(loss.item()))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        """

        # decay epsilon
        if self.ε > self.ε_min:
            self.ε *= self.ε_decay

        # Every update_interval, set target network's parameters to Q network's parameter
        time_to_update = (self.epoch % self.update_interval == 0)

        if time_to_update:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def workout(self):
        #ie. train the ENTIRE reinforcement learning model through one episode
        
        # reset epoch to 0
        self.epoch = 0
        self.timestep=0
        self.tradingEnvironment.inventory = [] # reset inventory
        # also reset experience replay buffer to zero (?)
        self.erBuffer = ExperienceReplayBuffer(self.batch_size)
        
        iteration_steps = self.tradingEnvironment.total_timesteps

        while self.timestep < iteration_steps-1: #-1 is here because we don't want to reach the terminal state as there is nothing to consider

            #print('Epoch ' + str(self.epoch) )
            
            # if erbuffer is empty (no steps taken yet), take the default values
            if self.epoch == 0:
                

                state = self.tradingEnvironment.zero_state
            else: #else take the last's next state in the buffer - which would give the current state
                state = self.erBuffer.buffer[-1]['next_state']
            
            self.step(state)

    def gymroutine(self, episodes):

        for i in range(episodes):
            self.workout()


        profits = sum([item['reward'] for item in self.erBuffer.buffer])
        print('Profits: '+ str(profits))

    def visualize(self):
        # visualizes trade actions in the most recent workout
        plt.plot(self.tradingEnvironment.price_data)

        actions_taken = [item['action_index'] for item in self.erBuffer.buffer]
        for i, value in enumerate(actions_taken):
            if value == 0:
                plt.plot(i, self.tradingEnvironment.price_data[i], marker='^', markersize=7, markerfacecolor='green', markeredgecolor='none')
            elif value == 1:
                plt.plot(i, self.tradingEnvironment.price_data[i], marker='v', markersize=7, markerfacecolor='red', markeredgecolor='none')
        

        plt.show()

    def trading_metrics(self):
        # gives metrics for the most recent workout

        bnh_return = (self.tradingEnvironment.price_data[-1] - self.tradingEnvironment.price_data[0])/self.tradingEnvironment.price_data[0]
        final_price, final_holdings, final_balance = self.erBuffer.buffer[-1]['next_state']
        final_portfolio = final_price * final_holdings + final_balance

        initial_price, initial_holdings, initial_balance = self.erBuffer.buffer[0]['state']
        initial_portfolio = initial_price * initial_holdings + initial_balance

        agent_return = (final_portfolio - initial_portfolio)/initial_portfolio

        print('BnH Return: ' + str(bnh_return))
        
        print('Agent Return: ' + str(agent_return))
    




    # implement a function that will take in how many training episodes and then does how many rounds of training. the idea is to get a reward convergence graph etc




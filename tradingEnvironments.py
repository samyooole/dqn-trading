import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

class purePriceEnvironment:

    def __init__(self, price_data, holdings, balance):
        self.price_data = price_data
        
        self.holdings = holdings #HWUC
        self.balance = balance #initial amount of cash available - as equities are bought, we draw down upon cash 

        self.actions = ['buy', 'sell', 'hold']

        self.trade_dict = {'buy': +1, 'sell': -1, 'hold': 0}

        self.states_dim = 3 # my own price, holdings, and balance --> can you also take into account other prices, other holdings?
        self.actions_dim = 3 # buy, sell or hold - in this order!

        self.total_timesteps = len(price_data)

        self.zero_state = torch.tensor([price_data[0], holdings, balance], dtype=torch.float32)

    
    def dynamics(self,this_state, trade, timestep): # move this off to tradingEnv
        # reward function is implicitly written here?
        
        this_price, this_holdings, this_balance = this_state

        # initial portfolio
        initial_portfolio = this_price*this_holdings + this_balance  # change if have more later on. HWUC

        next_holdings = this_holdings + self.trade_dict[trade] # increase in equity holding
        


        # Ensure next holdings and balance are non-negative
        if next_holdings < 0:
            next_holdings = 0
            next_balance = this_balance
        else:
            next_balance = this_balance - self.trade_dict[trade] * this_price # decrease in balance if I bought an equity

        next_price = self.price_data[timestep+1] # maybe add error handling here

        next_state = torch.tensor([next_price, next_holdings, next_balance], dtype=torch.float32)

        next_portfolio = next_price * next_holdings + next_balance
        reward = next_portfolio - initial_portfolio # implicit reward function here

        if next_balance <0:
            reward = -self.balance * 99999
        #if next_holdings <0:
        #    reward = -self.balance * 99999

        return reward, next_state


class ReturnEnvironment:
    # k sized window of past + today's returns
    # so state's dimensionality (in single equity case) is k returns (from last k days of experience) + 1 holdings + 1 balance = k+2
    # action dimensionality is still 3: buy, sell, hold (in this order)

    def __init__(self, price_data, holdings, balance, K=5):
        self.price_data = price_data
        self.inventory=[]

        self.returns_data = [(price_data[i] - price_data[i-1]) / price_data[i-1] for i in range(1, len(price_data))]
        
        #self.returns_data = [(price_data[i] - price_data[i-1]) for i in range(1, len(price_data))] # this is price difference, not returns

        self.K = K # HWUC if you want to customize

        windowed_returns = []
        for i in range(len(self.returns_data)-self.K+1):
            windowed_returns.append(self.returns_data[i:i+self.K]) # can only start from i+self.K

        self.windowed_returns = np.array(windowed_returns) # each row is a timestep, and each column is its K-1th away, ... 0th away return
        
        self.holdings = holdings #HWUC
        self.balance = balance #initial amount of cash available - as equities are bought, we draw down upon cash 

        self.actions = ['buy', 'sell', 'hold']

        self.trade_dict = {'buy': +1, 'sell': -1, 'hold': 0}

        self.states_dim = self.K + 2 # K windowed returns, holdings, and balance --> can you also take into account other prices, other holdings?
        self.actions_dim = 3 # buy, sell or hold - in this order!

        self.total_timesteps = len(self.price_data) - self.K


        self.zero_state = torch.tensor(list(self.windowed_returns[0]) + [self.holdings] + [self.balance], dtype=torch.float32)

    def dynamics(self, this_state, trade, timestep):
        this_holdings, this_balance = this_state[-2], this_state[-1]
        this_return_window = this_state[:self.K]
        this_price = self.price_data[timestep + self.K]

        
        if trade == 'buy':
            if this_balance.item() > this_price:
                reward = 0
                next_holdings = this_holdings + 1
                next_balance = this_balance - this_price
                self.inventory.append((1, this_price))
            else:
                reward = 0  # Penalize for insufficient balance
                next_holdings = this_holdings
                next_balance = this_balance

        elif trade == 'sell':
            if len(self.inventory) > 0:
                bought_qty, bought_price = self.inventory.pop(0)  # FIFO system
                profit = this_price - bought_price
                reward = profit
                next_holdings = this_holdings - 1
                next_balance = this_balance + this_price
            else:
                reward = 0  # Penalize for attempting to sell without inventory
                next_holdings = this_holdings
                next_balance = this_balance

        elif trade == 'hold':
            reward = 0
            next_holdings = this_holdings
            next_balance = this_balance

        next_state = torch.tensor(list(self.windowed_returns[timestep + 1]) + [next_holdings] + [next_balance], dtype=torch.float32)
        return reward, next_state

        # initial portfolio
        initial_portfolio = this_price*this_holdings + this_balance  # change if have more later on. HWUC

        next_holdings = this_holdings + self.trade_dict[trade] # increase in equity holding


        # Ensure next holdings and balance are non-negative
        if next_holdings < 0:
            next_holdings = 0
            next_balance = this_balance
            #reward = -self.balance*99999
            
            next_price = self.price_data[timestep+self.K+1] # maybe add error handling here
            next_portfolio = next_price * next_holdings + next_balance
            reward =next_portfolio - initial_portfolio

            next_state = torch.tensor(list(self.windowed_returns[timestep+1]) + [next_holdings] + [next_balance], dtype=torch.float32)
            return reward, next_state
        elif this_balance - self.trade_dict[trade] * this_price < 0:
            next_holdings=this_holdings
            next_balance=this_balance

            next_price = self.price_data[timestep+self.K+1]

            next_portfolio = next_price * next_holdings + next_balance
            reward =next_portfolio - initial_portfolio
            #reward=-self.balance*99999

            next_state = torch.tensor(list(self.windowed_returns[timestep+1]) + [next_holdings] + [next_balance], dtype=torch.float32)
            return reward, next_state
        else:
            next_balance = this_balance - self.trade_dict[trade] * this_price # decrease in balance if I bought an equity

            next_price = self.price_data[timestep+self.K+1] # maybe add error handling here

            next_state = torch.tensor(list(self.windowed_returns[timestep+1]) + [next_holdings] + [next_balance], dtype=torch.float32)

            next_portfolio = next_price * next_holdings + next_balance
            reward = next_portfolio - initial_portfolio # implicit reward function here
            #reward = (next_portfolio-initial_portfolio)/initial_portfolio

            #if next_balance <0:
            #    reward = -self.balance * 99999
            #if next_holdings <0:
            #    reward = -self.balance * 99999

            return reward, next_state

        

        
    
    
        
    
    def visualize(self, generated_buffer):
        # visualizes trade actions in the most recent workout
        plt.plot(self.price_data)

        actions_taken = [item['action_index'] for item in generated_buffer]
        for i, value in enumerate(actions_taken):
            if value == 0:
                plt.plot(i+self.K, self.price_data[i+self.K], marker='^', markersize=7, markerfacecolor='green', markeredgecolor='none')
            elif value == 1:
                plt.plot(i+self.K, self.price_data[i+self.K], marker='v', markersize=7, markerfacecolor='red', markeredgecolor='none')
        

        plt.show()

    def trading_metrics(self, generated_buffer):
        # gives metrics for the most recent workout

        bnh_return = (self.price_data[-1] - self.price_data[self.K])/self.price_data[self.K]
        final_price = self.price_data[-1]
        final_holdings = generated_buffer[-1]['next_state'][-2]
        final_balance = generated_buffer[-1]['next_state'][-1]
        final_portfolio = final_price * final_holdings + final_balance

        initial_price = self.price_data[self.K]
        initial_holdings = generated_buffer[self.K]['state'][-2]
        
        initial_balance = generated_buffer[self.K]['state'][-1]
        initial_portfolio = initial_price * initial_holdings + initial_balance

        agent_return = (final_portfolio - initial_portfolio)/initial_portfolio

        print('BnH Return: ' + str(bnh_return))
        
        print('Agent Return: ' + str(agent_return))
    


class AlbertReturnEnvironment:
    # k sized window of past + today's returns
    # so state's dimensionality (in single equity case) is k returns (from last k days of experience) + 1 holdings + 1 balance = k+2
    # action dimensionality is still 3: buy, sell, hold (in this order)

    def __init__(self, price_data, holdings, balance, K=5):
        self.price_data = price_data
        self.inventory=[]

        self.returns_data = [(price_data[i] - price_data[i-1]) / price_data[i-1] for i in range(1, len(price_data))]
        
        #self.returns_data = [(price_data[i] - price_data[i-1]) for i in range(1, len(price_data))] # this is price difference, not returns

        self.K = K # HWUC if you want to customize

        windowed_returns = []
        for i in range(len(self.returns_data)-self.K+1):
            windowed_returns.append(self.returns_data[i:i+self.K]) # can only start from i+self.K

        self.windowed_returns = np.array(windowed_returns) # each row is a timestep, and each column is its K-1th away, ... 0th away return
        
        self.holdings = holdings #HWUC
        self.balance = balance #initial amount of cash available - as equities are bought, we draw down upon cash 

        self.actions = ['buy', 'sell', 'hold']

        self.trade_dict = {'buy': +1, 'sell': -1, 'hold': 0}

        self.states_dim = self.K + 3 # K windowed returns, price, holdings, and balance --> can you also take into account other prices, other holdings?
        self.actions_dim = 3 # buy, sell or hold - in this order!

        self.total_timesteps = len(self.price_data) - self.K


        self.zero_state = torch.tensor(list(self.windowed_returns[0]) + [self.price_data[self.K]] + [self.holdings] + [self.balance], dtype=torch.float32)

    def dynamics(self, this_state, trade, timestep):
        this_holdings, this_balance = this_state[-2], this_state[-1]
        this_return_window = this_state[:self.K]
        this_price = this_state[-3]

        
        if trade == 'buy':
            if this_balance.item() > this_price:
                reward = 0
                next_holdings = this_holdings + 1
                next_balance = this_balance - this_price
                self.inventory.append((1, this_price))
                action_override = 'buy'
            else:
                reward = 0  # Penalize for insufficient balance
                next_holdings = this_holdings
                next_balance = this_balance
                action_override = 'hold'

        elif trade == 'sell':
            if len(self.inventory) > 0:
                bought_qty, bought_price = self.inventory.pop(0)  # FIFO system
                profit = this_price - bought_price
                reward = profit
                next_holdings = this_holdings - 1
                next_balance = this_balance + this_price
                action_override = 'sell'
            else:
                reward = 0  # Penalize for attempting to sell without inventory
                next_holdings = this_holdings
                next_balance = this_balance
                action_override = 'hold'
        elif trade == 'hold':
            reward = 0
            next_holdings = this_holdings
            next_balance = this_balance
            action_override = 'hold'

        # penalty for holding in balance
            
        reward =- next_balance*7.43278783121859e-05

        # unrealized profit also as reward
        
        next_price = self.price_data[timestep + self.K + 1]
        reward=+ next_price*next_holdings + next_balance - (self.price_data[self.K] * self.holdings + self.balance)

        next_state = torch.tensor(list(self.windowed_returns[timestep + 1]) + [next_price] + [next_holdings] + [next_balance], dtype=torch.float32)
        #print('Return so far: ' + str(next_price * next_holdings + next_balance - self.balance))
        return reward, next_state, action_override

        # initial portfolio
        initial_portfolio = this_price*this_holdings + this_balance  # change if have more later on. HWUC

        next_holdings = this_holdings + self.trade_dict[trade] # increase in equity holding


        # Ensure next holdings and balance are non-negative
        if next_holdings < 0:
            next_holdings = 0
            next_balance = this_balance
            #reward = -self.balance*99999
            
            next_price = self.price_data[timestep+self.K+1] # maybe add error handling here
            next_portfolio = next_price * next_holdings + next_balance
            reward =next_portfolio - initial_portfolio

            next_state = torch.tensor(list(self.windowed_returns[timestep+1]) + [next_holdings] + [next_balance], dtype=torch.float32)
            return reward, next_state
        elif this_balance - self.trade_dict[trade] * this_price < 0:
            next_holdings=this_holdings
            next_balance=this_balance

            next_price = self.price_data[timestep+self.K+1]

            next_portfolio = next_price * next_holdings + next_balance
            reward =next_portfolio - initial_portfolio
            #reward=-self.balance*99999

            next_state = torch.tensor(list(self.windowed_returns[timestep+1]) + [next_holdings] + [next_balance], dtype=torch.float32)
            return reward, next_state
        else:
            next_balance = this_balance - self.trade_dict[trade] * this_price # decrease in balance if I bought an equity

            next_price = self.price_data[timestep+self.K+1] # maybe add error handling here

            next_state = torch.tensor(list(self.windowed_returns[timestep+1]) + [next_holdings] + [next_balance], dtype=torch.float32)

            next_portfolio = next_price * next_holdings + next_balance
            reward = next_portfolio - initial_portfolio # implicit reward function here
            #reward = (next_portfolio-initial_portfolio)/initial_portfolio

            #if next_balance <0:
            #    reward = -self.balance * 99999
            #if next_holdings <0:
            #    reward = -self.balance * 99999

            return reward, next_state

        

        
    
    
        
    
    def visualize(self, generated_buffer):
        # visualizes trade actions in the most recent workout
        plt.plot(self.price_data)

        actions_taken = [item['action_index'] for item in generated_buffer]
        for i, value in enumerate(actions_taken):
            if value == 0:
                plt.plot(i+self.K, self.price_data[i+self.K], marker='^', markersize=7, markerfacecolor='green', markeredgecolor='none')
            elif value == 1:
                plt.plot(i+self.K, self.price_data[i+self.K], marker='v', markersize=7, markerfacecolor='red', markeredgecolor='none')
        

        plt.show()

    def trading_metrics(self, generated_buffer):
        # gives metrics for the most recent workout

        bnh_return = (self.price_data[-1] - self.price_data[self.K])/self.price_data[self.K]
        final_price = self.price_data[-1]
        final_holdings = generated_buffer[-1]['next_state'][-2]
        final_balance = generated_buffer[-1]['next_state'][-1]
        final_portfolio = final_price * final_holdings + final_balance

        initial_price = self.price_data[self.K]
        initial_holdings = generated_buffer[self.K]['state'][-2]
        
        initial_balance = generated_buffer[self.K]['state'][-1]
        initial_portfolio = initial_price * initial_holdings + initial_balance

        agent_return = (final_portfolio - initial_portfolio)/initial_portfolio

        print('BnH Return: ' + str(bnh_return))
        
        print('Agent Return: ' + str(agent_return))
    


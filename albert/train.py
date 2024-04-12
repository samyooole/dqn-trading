import yfinance as yf
from dqnTrader import DQNTrader
import numpy as np
from utils import *
import torch


model_name = 'DQN'
#stock_name = '^GSPC_2018'
window_size = 10
num_episode = 10
initial_balance = 50000


spy = yf.download('SPY', start='2014-01-01', end='2015-01-01')
spy = spy['Adj Close']
spy = list(spy)

stock_prices = spy
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select learning model
agent = DQNTrader(state_dim=window_size + 3, balance=initial_balance)

def hold(actions, t):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions.detach())[0][1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            actions[0][next_probable_action] = 1 # reset this action's value to the highest
            return 'Hold', actions

def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        agent.buy_dates.append(t)
        return 'Buy: ${:.2f}'.format(stock_prices[t])

def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        agent.sell_dates.append(t)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)

def workout(num_episode):
    num_experience_replay=0
    for e in range(1, num_episode + 1):

        agent.reset() # reset to initial balance and hyperparameters
        state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

        for t in range(1, trading_period + 1):
            
            reward = 0
            next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
            previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

            actions = agent.model(torch.FloatTensor(state))
            action = agent.act(state)
            
            # execute position
            if action == 0: # hold
                execution_result = hold(actions, t)
            if action == 1: # buy
                execution_result = buy(t)      
            if action == 2: # sell
                execution_result = sell(t)        
            
            # check execution result
            if execution_result is None:
                reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
            else:
                if isinstance(execution_result, tuple): # if execution_result is 'Hold'
                    actions = execution_result[1]
                    execution_result = execution_result[0]     

            # calculate reward
            current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
            unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
            reward += unrealized_profit

            agent.portfolio_values.append(current_portfolio_value)
            agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

            done = True if t == trading_period else False
            agent.remember(state, actions, reward, next_state, done)

            # update state
            state = next_state

            # experience replay
            if len(agent.memory) > agent.buffer_size:
                num_experience_replay += 1
                loss = agent.experience_replay()
                print('Loss: ' + str(loss) )

            if done:
                portfolio_return = evaluate_portfolio_performance(agent)
                returns_across_episodes.append(portfolio_return)

            portfolio_return = evaluate_portfolio_performance(agent)
            print(portfolio_return)


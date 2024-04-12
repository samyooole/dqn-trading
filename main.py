from dqnTrader import dqnTrader
import tradingEnvironments
from tradingEnvironments import purePriceEnvironment
import yfinance as yf
from importlib import reload

tradingEnvironments = reload(tradingEnvironments)
ReturnEnvironment = tradingEnvironments.ReturnEnvironment

AlbertReturnEnvironment = tradingEnvironments.AlbertReturnEnvironment




spy = yf.download('SPY', start='2022-01-01', end='2023-01-01')
spy = spy['Adj Close']
spy = list(spy)


aapl = yf.download('AAPL', start='2021-01-01', end='2022-04-30')
aapl = aapl['Adj Close']
aapl = list(aapl)




stockenv = AlbertReturnEnvironment(spy, holdings=0,balance=10000, K=20)


mydqn = dqnTrader( tradingEnvironment=stockenv,  training_interval = 15, update_interval = 40, γ=0.99 ,batch_size = 40,learning_rate=0.1) # this is good



mydqn.gymroutine(episodes=1)
mydqn.gymroutine(episodes=20)

while True:
    mydqn.gymroutine(episodes=5) # 7 * 30 for 10000
    print(mydqn.q_network(stockenv.zero_state))
mydqn.ε


mydqn.tradingEnvironment = AlbertReturnEnvironment(aapl_test, holdings=0,balance=10000, K=1)

stockenv.visualize(mydqn.erBuffer.buffer)
stockenv.trading_metrics(mydqn.erBuffer.buffer)

mydqn.visualize()
mydqn.trading_metrics()










### SEAL THIS UP: aapl worked!


# why randomly it will stumble onto a good starting parameter configuration



stockenv = AlbertReturnEnvironment(spy, holdings=0,balance=10000, K=1)


mydqn = dqnTrader( tradingEnvironment=stockenv,  training_interval = 15, update_interval = 40, γ=0.95 ,batch_size = 40,learning_rate=0.01) # this is good



mydqn.gymroutine(episodes=1)
mydqn.gymroutine(episodes=10)










###

stockenv = purePriceEnvironment(spy, holdings=30, balance=4000)
mydqn = dqnTrader( tradingEnvironment=stockenv,  training_interval = 30, update_interval = 90, γ=0.98 ,batch_size = 30,learning_rate=0.001)

mydqn.gymroutine(episodes=10)























"""
self.trading_metrics()
self.visualize()

[item['action_index'] for item in self.erBuffer.buffer]

self.workout()
profits = sum([item['reward'] for item in self.erBuffer.buffer])
profits

self.erBuffer.buffer
"""
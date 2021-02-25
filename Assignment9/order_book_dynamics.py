from rl.chapter9.order_book import OrderBook, DollarsAndShares, PriceSizePairs
from rl.markov_process import MarkovProcess
from rl.distribution import Distribution, SampledDistribution
from typing import Optional
from numpy.random import poisson

class OrderBookDynamics(MarkovProcess):

    bid_mkt_distr: Distribution[int]  # distribution of market bid orders
    bid_limit_distr: Distribution[float, int] # distribution of limit bid orders
    ask_mkt_distr: Distribution[int]
    ask_limit_distr: Distribution[float, int]
    bid_limit_num: Distribution[int]  # distribution of number of limit bid orders, might have different shares and prices
    ask_limit_num: Distribution[int]

    def transition(self, state: OrderBook) -> Optional[Distribution[OrderBook]]:
        '''Given a state of the process, returns a distribution of
        the next states.  Returning None means we are in a terminal state.
        '''
        def sampler_func(state: OrderBook):
            next_state = state.buy_market_order(self.bid_mkt_distr.sample())[1]
            next_state = next_state.sell_market_order(self.ask_mkt_distr.sample())[1]
            for i in range(self.bid_limit_num.sample()):
                next_state.buy_limit_order(self.bid_limit_distr.sample()[0], self.bid_limit_distr.sample()[1])
            for i in range(self.ask_limit_num.sample()):
                next_state.sell_limit_order(self.ask_limit_distr.sample()[0], self.ask_limit_distr.sample()[0])
            return next_state

        return SampledDistribution(sampler_func, expectation_samples=1000)


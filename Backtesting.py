import pandas as pd
import numpy as np

# names for the strategy selection
NAMES = ['3M_Momo', '6M_Momo', '12M_Momo', '3M_LowVol' , '6M_LowVol']
# allocation algorithm names (equally weighted so far ##ADD MORE##)
ALLOCATION = ['EW']

class Backtest:

    def __init__(self, returns, calendar, strategy, N):
        """Main class to perform the computations.
        Parameters
        ----------
        returns: pandas.DataFrame
            Matrix of returns. There must be no zeros
            nor Not a Number.
        calendar: pandas.DatetimeIndex
            The vector of dates to simulate.
            The simulation starts one year after.
        strategy: dictionary
            With keys 'name' (strategy algorithm)
            and 'allocation' (weight distribution algorithm).
        N: int
            Number of assets in your portfolio
        """

        self.N = N

        # we need to know the type of the strategy
        if not isinstance(strategy, dict):
            raise ValueError("The strategy is a dictionary.")
        if 'name' not in strategy.keys():
            raise ValueError("Specify the strategy name!")
        if 'allocation' not in strategy.keys():
            raise ValueError("Specify the weight distribution algorithm!")
        if strategy['name'] not in NAMES:
            raise NotImplementedError("Your stock picking algorithm seems to "
                                      "be not implemented yet.")
        if strategy['allocation'] not in ALLOCATION:
            raise NotImplementedError("Your weight distribution algorithm seems to "
                                      "be not implemented yet.")
        self.strategy = strategy

        # we need the returns of the assets
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("The passed matrix is not a pandas.DataFrame!")
        else:
            if not isinstance(returns.index, pd.DatetimeIndex):
                raise ValueError("The passed matrix has not the proper index.")

        self.dates = returns.index
        self.returns = returns.values

        self.calendar = calendar
        self.weights = None
        self._aux_cum_return = None
        self.selected = None

    @property
    def cumulative_return(self):
        x = pd.DataFrame(data=self._aux_cum_return,
                         index=self.dates,
                         columns=['Cumulative Return'])
        x = x.replace(0, np.nan).dropna()
        return x

    @staticmethod
    def evolve_weights(returns, weights):
        num = weights * (1 + returns)
        return np.nan_to_num(num / np.sum(num))

    def run(self):

        if self.weights is not None or self._aux_cum_return:
            raise ValueError("You already simulated!")

        dates = self.dates

        # initialize the simulation
        self.weights = np.zeros((len(dates), self.returns.shape[1]))
        self._aux_cum_return = np.zeros((len(dates), 1))

        # we start iterating!
        for i, day in enumerate(dates):

            # we need at least 261 days of history
            if i > 252:

                today = i  # today's index
                yty = today - 1  # yesterday's index

                rets = self.returns[today]  # today's asset returns

                if today > 0:

                    # evolve the weights of my portfolio
                    self.weights[today, :] = \
                        self.evolve_weights(rets, self.weights[yty, :])

                # we run our algorithm and do the trades
                if day in self.calendar:

                    print(str(day)[:-9])

                    self.strategy_selection(today)
                    self.asset_allocation(today)

                # update the cumulative return of the portfolio
                if day > dates[0]:
                    # calculate return of the portfolio
                    rent = np.sum(self.weights[yty] * rets)

                    # accumulate equity
                    self._aux_cum_return[today] = \
                        (1 + rent) * (1 + self._aux_cum_return[yty]) - 1

    def strategy_selection(self, today):

        # prepare the data
        if self.strategy['name'] == '3M_Momo':
            a = self.returns[today - 21 * 3:today, :]
        elif self.strategy['name'] == '6M_Momo':
            a = self.returns[today - 21 * 6:today, :]
        elif self.strategy['name'] == '12M_Momo':
            a = self.returns[today - 21 * 12:today, :]
        elif self.strategy['name'] == '3M_LowVol':
            a = self.returns[today - 21 * 3:today, :]
        elif self.strategy['name'] == '6M_LowVol':
            a = self.returns[today - 21 * 6:today, :]

        # do the strategy picking
        if self.strategy['name'] == '3M_Momo' or \
            self.strategy['name'] == '6M_Momo' or \
            self.strategy['name'] == '12M_Momo':

            # get the cumulative return of the assets
            a = np.cumprod(1 + a, 0) - 1
            # take the last data point
            value = a[-1, :]
            # these are the selected
            self.selected = \
                value > np.percentile(value, (1 - self.N / len(value)) * 100)
        elif self.strategy['name'] == '3_LowVol' or \
              self.strategy['name'] == '6M_LowVol':
            # calculate the annualised volatility
            std = np.std(a, 0) * np.sqrt(252)
            self.selected = std < np.percentile(std, (self.N / len(std)) * 100)

    def asset_allocation(self, today):

        # do equally weight
        if self.strategy['allocation'] == 'EW':
            self.weights[today] = 0
            self.weights[today, self.selected] = 1 / self.N

if __name__ == '__main__':

    # this is the date range we want to simulate
    dates = pd.bdate_range('20000101', '20180831')
    # let's generate some random normal returns for illustrative purposes
    random_returns = pd.DataFrame(data=np.random.normal(0, 1, size=(len(dates), 500)) / 100,
                                  index=dates)
    # we create the execution dates -- where we trade assets
    calendar = dates[dates.is_month_end]
    # what kind of strategy do we want?
    strategy = {'name': '12M_Momo',
                'allocation': 'EW'}
    # construct the class and run the backtest
    backtest = Backtest(random_returns, calendar, strategy, 50)
    backtest.run()
    # plot the result!
    backtest.cumulative_return.plot()
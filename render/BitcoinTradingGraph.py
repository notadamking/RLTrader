

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
from pandas.plotting import register_matplotlib_converters

style.use('ggplot')
register_matplotlib_converters()

VOLUME_CHART_HEIGHT = 0.33


class BitcoinTradingGraph:
    """A Bitcoin trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df):
        self.df = df
        self.df['Time'] = self.df['Date'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d %I-%p'))
        self.df = self.df.sort_values('Time')

        # Create a figure on screen and set the title
        self.fig = plt.figure()

        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1)

        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_net_worth(self, current_step, net_worths, dates):
        # Clear the frame rendered last step
        self.net_worth_ax.clear()

        # Plot net worths
        self.net_worth_ax.plot(
            dates, net_worths, label='Net Worth', color="g")

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = self.df['Time'].values[current_step]
        last_net_worth = net_worths[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(last_net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.net_worth_ax.set_ylim(
            min(net_worths) / 1.25, max(net_worths) * 1.25)

    def _render_price(self, current_step, net_worths, dates):
        self.price_ax.clear()

        # Plot price using candlestick graph from mpl_finance
        self.price_ax.plot(
            self.df['Time'].values[:current_step + 1], self.df['Close'].values[:current_step + 1], color="black")

        last_date = self.df['Time'].values[current_step]
        last_close = self.df['Close'].values[current_step]
        last_high = self.df['High'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, current_step, net_worth, dates):
        self.volume_ax.clear()

        volume = np.array(self.df['Volume BTC'].values[:current_step + 1])

        self.volume_ax.plot(dates, volume,  color='blue')
        self.volume_ax.fill_between(dates, volume, color='blue')

        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, current_step, trades):
        for trade in trades:
            if trade['step'] in range(current_step):
                date = self.df['Time'].values[trade['step']]
                close = self.df['Close'].values[trade['step']]

                if trade['type'] == 'buy':
                    color = 'g'
                else:
                    color = 'r'

                self.price_ax.annotate(' ', (date, close),
                                       xytext=(date, close),
                                       size="large",
                                       arrowprops=dict(arrowstyle='simple', facecolor=color))

    def render(self, current_step, net_worths, trades):
        net_worth = round(net_worths[-1], 2)
        profit_percent = round(
            (net_worth - net_worths[0]) / net_worths[0] * 100, 2)
        self.fig.suptitle(
            'Net worth: $' + str(net_worth) + ' | Profit: ' + str(profit_percent) + '%')

        dates = self.df['Time'].values[:current_step + 1]

        self._render_net_worth(current_step, net_worths, dates)
        self._render_price(current_step, net_worths, dates)
        self._render_volume(current_step, net_worths, dates)
        self._render_trades(current_step, trades)

        date_labels = self.df['Date'].values[:current_step + 1]

        self.price_ax.set_xticklabels(
            date_labels, rotation=45, horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()

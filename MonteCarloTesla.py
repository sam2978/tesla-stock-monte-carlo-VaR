import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock_data = pd.read_csv('TeslaStockData.csv', parse_dates=True, index_col='Date')
stocksfrom3000 = stock_data.iloc[3000:] #narrowed the data set since starting from 2010 made simulation less accurate
closeprices = stocksfrom3000['Close']

sims = 1000
forecast_days = 252

#create simulation array with all 0s
simulated_prices = np.zeros((forecast_days, sims))

last_price = closeprices.iloc[-1]

#calculate daily returns and remove null values
returns = closeprices.pct_change().dropna()


for i in range(sims):
    #find random daily returns and create cumulative return series
    cumulative_return = np.random.choice(returns, size=forecast_days, replace=True).cumsum()
    #get sims with cumulative returns
    simulated_prices[:, i] = last_price * (1 + cumulative_return)

#final sumulated prices
final_prices = simulated_prices[-1, :]

#above and below last price counts
num_above = np.sum(final_prices > last_price)
num_below = np.sum(final_prices < last_price)

#probability of profit
prob_profit = num_above / sims

#mean of all sims
expected_final_price = np.mean(final_prices)

#expected return percentage
expected_return_pct = (expected_final_price / last_price - 1) * 100

#Value at Risk 95% - the price point where we can be 95% certain losses won't exceed
#VAR Calculation
final_returns = (final_prices / last_price) - 1
#VAR percentile calculation
conf_level = 0.95
alpha = 1 - conf_level
#5th percentile of returns is the loss threshold for 95% confidence level
pctile = np.percentile(final_returns, 100 * alpha) #find 5th percentile of returns
var_pct = -pctile if pctile < 0 else 0.0 #positive value for VaR
var_dollar_loss = var_pct * last_price #dollar loss at VaR level
# price level corresponding to VaR (the price you'd be at the pctile)
var_price = last_price * (1 + pctile)

#print(f"VaR ({int(conf_level*100)}%) = {var_pct:.2%} loss -> about ${var_abs:.2f} on ${last_price:.2f}")

print(f"VaR ({int(conf_level*100)}%) = {var_pct:.2%} loss") #value at risk percentage
print(f"VaR price level = ${var_price:.2f}  (i.e. about ${var_dollar_loss:.2f} loss from ${last_price:.2f})") #actual 5th percentile price


#plot the simulations
plt.figure(figsize=(10,6))
plt.plot(simulated_prices, color='lightblue')
plt.axhline(y=last_price, color='r', linestyle='--', label='Last Actual Price')
plt.axhline(y=var_price, color='orange', linestyle='--', label='VaR 95% Level')
plt.axhline(y=expected_final_price, color='g', linestyle='--', label='Expected Final Price')
plt.title('Monte Carlo Simulation of Tesla Stock Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

print("Last Actual Price: ${:.2f}".format(last_price))
print("Expected Final Price after {} days: ${:.2f}".format(forecast_days, expected_final_price))
print("Expected Return Percentage: {:.2f}%".format(expected_return_pct))
print("Probability of Profit: {:.2f}%".format(prob_profit * 100))
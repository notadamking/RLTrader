# Bitcoin-Trader-RL

In this series of articles, we've created and optimized a Bitcoin trading agent to be highly profitable using deep reinforcement learning.

Discord server: https://discord.gg/ZZ7BGWh

Data sets: https://www.cryptodatadownload.com/data/northamerican/

If you'd like to learn more about how we created this agent, check out the Medium article: https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29

Later, we optimized this repo for massive profits using feature engineering and Bayesian optimization, check it out:
https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b

# Getting Started

The first thing you will need to do to get started is install the requirements in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The requirements include the `tensorflow-gpu` library, though if you do not have access to a GPU, you should replace this requirement with `tensorflow`.

# Optimizing, Training, and Testing

While you could just let the agent train and run with the default PPO2 hyper-parameters, your agent would likely not be very profitable. The `stable-baselines` library provides a great set of default parameters that work for most problem domains, but we need to better.

To do this, you will need to run `optimize.py`. Within the file, you can define the `reward_strategy` for the environment to use, this is currently defaulted to `sortino`.

```bash
python ./optimize.py
```

This can take a while (hours to days depending on your hardware setup), but over time it will print to the console as trials are completed. Once a trial is completed, it will be stored in `./data/params.db`, an SQLite database, from which we can pull hyper-parameters to train our agent.

From there, you can train an agent with the best set of hyper-parameters, and later test it on completely new data to verify the generalization of the algorithm.

# Project Roadmap

If you would like to contribute, here is the roadmap for the future of this project. To assign yourself to an item, please create an Issue/PR titled with the item from below and I will add your name to the list.

## Stage 0:
* Create a generic data loader for inputting multiple data sources (.csv, API, in-memory, etc.) **[sph3rex, @lukeB]**
  * Map each data source to OHCLV format w/ same date/time format
* Implement live trading capabilities
  * Allow model/agent to be passed in at run time
  * Allow live data to be saved in a format that can be later trained on
  * Enable paper-trading by default
* Enable complete multi-processing throughout the environment
  * Optionally replace SQLite db with Postgres to enable multi-processed Optuna training
  * Replace `DummyVecEnv` with `SubProcVecEnv` everywhere throughout the code
* Find source of CPU bottlenecks to improve GPU utilization
  * Improve speed of pandas methods by taking advantage of GPU
  * Pre-process any data that is not currently being pre-processed
* Find source of memory leak (in `RLTrader.optimize`) and squash it
  
## Stage 1:
* Allow features to be added/removed at runtime
  * Create simple API for turning off default features (e.g. prediction, indicators, etc.)
  * Create simple API for adding new features to observation space
* Add more optional features to the feature space
  * Other exchange pair data (e.g. LTC/USD, ETH/USD, EOS/BTC, etc.)
  * Twitter sentiment analysis
  * Google trends analysis
  * Order book data
  * Market tick data
* Create a generic prediction interface to allow any prediction function to be used
  * Implement SARIMAX using generic interface
  * Implement FB Prophet using generic interface
  * Implement pre-trained LSTM using generic interface
* Allow trained models to be saved to a local database (SQLite/Postgres)
  * Save performance metrics with the model

  

# Contributing

Contributions are encouraged and I will always do my best to get them implemented into the library ASAP. This project is meant to grow as the community around it grows. Let me know if there is anything that you would like to see in the future or if there is anything you feel is missing.

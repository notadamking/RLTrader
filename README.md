# Bitcoin-Trader-RL

In this series of articles, we've created and optimized a Bitcoin trading agent to be highly profitable using deep reinforcement learning.

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
 
 # Finding Hyper-Parameters
 
While you could just let the agent train and run with the default PPO2 hyper-parameters, your agent would likely not be very profitable. The `stable-baselines` library provides a great set of default parameters that work for most problem domains, but  we need to better.

To do this, you will need to run `optimize.py`. Within the file, you can define the `reward_strategy` for the environment to use, this is currently defaulted to `sortino`.

```bash
python ./optimize.py
```

This will take a while (hours to days depending on your hardware setup), but over time it will print to the console as trials are completed. Once a trial is completed, it will be stored in `./params.db`, an SQLite database, from which we can pull hyper-parameters to train our agent.

# Training Agents

Once you've found a good set of hyper-parameters, we can train an agent with that set. To do this, you will want to open `train.py` and ensure the `reward_strategy` is set to the correct strategy. Then let `train.py` run until you've got some saved models to test.

```bash
python ./train.py
```

If you have already trained a model, and would like to resume training from the next epoch, you can set `curr_idx` at the top of the file to the index of the last trained model. Otherwise, leave this at `-1` to start training at epoch 0.

# Testing Agents

Once you've successfully trained and saved a model, it's time to test it. Open up `test.py` and set the `reward_strategy` to the correct strategy and `curr_idx` to the index of the agent you'd like to train. Then run `test.py` to watch your agent trade.

```bash
python ./test.py
```

# Contributing

Contributions are encouraged and I will always do my best to get them implemented into the library ASAP. This project is meant to grow as the community around it grows. Let me know if there is anything that you would like to see in the future or if there is anything you feel is missing.

# RLTrader (Formerly Bitcoin-Trader-RL)

[![Build Status](https://travis-ci.org/notadamking/RLTrader.svg?branch=master)](https://travis-ci.org/notadamking/RLTrader)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![GPL License](https://img.shields.io/github/license/notadamking/RLTrader.svg?color=brightgreen)](https://opensource.org/licenses/GPL-3.0/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Github Stars](https://img.shields.io/github/stars/notadamking/RLTrader.svg)](https://github.com/notadamking/RLTrader)

In this series of articles, we've created and optimized a Bitcoin trading agent to be highly profitable using deep reinforcement learning.

Discord server: https://discord.gg/ZZ7BGWh

Data sets: https://www.cryptodatadownload.com/data/northamerican/

If you'd like to learn more about how we created this agent, check out the Medium article: https://towardsdatascience.com/creating-bitcoin-trading-bots-that-dont-lose-money-2e7165fb0b29

Later, we optimized this repo using feature engineering, statistical modeling, and Bayesian optimization, check it out:
https://towardsdatascience.com/using-reinforcement-learning-to-trade-bitcoin-for-massive-profit-b69d0e8f583b

![Live trading visualization](https://github.com/notadamking/RLTrader/blob/master/visualization.gif)

# Getting Started

The first thing you will need to do to get started is install the requirements in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The requirements include the `tensorflow-gpu` library, though if you do not have access to a GPU, you should replace this requirement with `tensorflow`.

# Optimizing, Training, and Testing

While you could just let the agent train and run with the default PPO2 hyper-parameters, your agent would likely not be very profitable. The `stable-baselines` library provides a great set of default parameters that work for most problem domains, but we need to better.

To do this, you will need to run `optimize.py`.

```bash
python ./optimize.py
```

This can take a while (hours to days depending on your hardware setup), but over time it will print to the console as trials are completed. Once a trial is completed, it will be stored in `./data/params.db`, an SQLite database, from which we can pull hyper-parameters to train our agent.

From there, you can train an agent with the best set of hyper-parameters, and later test it on completely new data to verify the generalization of the algorithm.

# Project Roadmap

If you would like to contribute, here is the roadmap for the future of this project. To assign yourself to an item, please create an Issue/PR titled with the item from below and I will add your name to the list.

## Stage 0:

- ~Create a generic data loader for inputting multiple data sources (.csv, API, in-memory, etc.)~ **[@sph3rex, @lukeB, @notadamking]** :white_check_mark:
  - ~Map each data source to OHCLV format w/ same date/time format~ \*\*[@notadamking] :white_check_mark:
- Implement live trading capabilities **[@notadamking]**
  - Allow model/agent to be passed in at run time **[@notadamking]**
  - Allow live data to be saved in a format that can be later trained on **[@notadamking]**
  - Enable paper-trading by default **[@notadamking]**
- Enable complete multi-processing throughout the environment
  - Optionally replace SQLite db with Postgres to enable multi-processed Optuna training
  - Replace `DummyVecEnv` with `SubProcVecEnv` everywhere throughout the code
- Find source of CPU bottlenecks to improve GPU utilization
  - Improve speed of pandas methods by taking advantage of GPU
  - Pre-process any data that is not currently being pre-processed
- Find source of memory leak (in `RLTrader.optimize`) and squash it
- Allow features to be added/removed at runtime
  - Create simple API for turning off default features (e.g. prediction, indicators, etc.)
  - Create simple API for adding new features to observation space
- Add more optional features to the feature space
  - Other exchange pair data (e.g. LTC/USD, ETH/USD, EOS/BTC, etc.)
  - Twitter sentiment analysis
  - Google trends analysis
  - Order book data
  - Market tick data
- Create a generic prediction interface to allow any prediction function to be used
  - Implement SARIMAX using generic interface
  - Implement FB Prophet using generic interface
  - Implement pre-trained LSTM using generic interface
- Allow trained models to be saved to a local database (SQLite/Postgres)
  - Save performance metrics with the model

## Stage 1:

- Implement a Generative Aderversarial Network (GAN) for accurately simulating asset price fluctuations
  - Implement Monte Carlo rollouts to find the most probabilistic outcomes
- Implement a custom RL agent using ODEs or other state-of-the-art algorithm (relational recurrent networks)
  - Incorporate GAN predictions into model state
- Implement `xgboost` and Stacked Auto-encoders to improve the feature selection of the model
- Experiment with Auto-decoders to remove noise from the observation space
- Implement self-play in a multi-process environment to improve model exploration

  - Experiment with dueling actors vs tournament of dueling agents

# Contributing

Contributions are encouraged and I will always do my best to get them implemented into the library ASAP. This project is meant to grow as the community around it grows. Let me know if there is anything that you would like to see in the future or if there is anything you feel is missing.

**Working on your first Pull Request?** You can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github)

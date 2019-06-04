import gym
import optuna
import pandas as pd
import pickle

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators


df = pd.read_csv('./data/coinbase_hourly.csv')
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

test_len = int(len(df) * 0.2)
train_len = int(len(df)) - test_len

test_df = df[train_len:]

profit_study = optuna.load_study(study_name='ppo2_profit',
                          storage='sqlite:///params.db')
profit_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    test_df, reward_func="profit", forecast_len=int(profit_study.best_trial.params['forecast_len']), confidence_interval=profit_study.best_trial.params['confidence_interval'])])

sortino_study = optuna.load_study(study_name='ppo2_sortino',
storage='sqlite:///params.db')
sortino_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    test_df, reward_func="profit", forecast_len=int(sortino_study.best_trial.params['forecast_len']), confidence_interval=sortino_study.best_trial.params['confidence_interval'])])

# calmar_study = optuna.load_study(study_name='ppo2_sortino',
# storage='sqlite:///params.db')
# calmar_env = DummyVecEnv([lambda: BitcoinTradingEnv(
#    test_df, reward_func="profit", forecast_len=int(calmar_study.best_trial.params['forecast_len']), confidence_interval=calmar_study.best_trial.params['confidence_interval'])])

omega_study = optuna.load_study(study_name='ppo2_omega',
storage='sqlite:///params.db')
omega_env = DummyVecEnv([lambda: BitcoinTradingEnv(
    test_df, reward_func="profit", forecast_len=int(omega_study.best_trial.params['forecast_len']), confidence_interval=omega_study.best_trial.params['confidence_interval'])])


profit_model = PPO2.load('./agents/ppo2_profit_4.pkl', env=profit_env)
sortino_model = PPO2.load('./agents/ppo2_sortino_4.pkl', env=sortino_env)
# calmar_model = PPO2.load('./agents/ppo2_calmar_4.pkl', env=calmar_env)
omega_model = PPO2.load('./agents/ppo2_omega_4.pkl', env=omega_env)

profit_obs = profit_env.reset()
sortino_obs = sortino_env.reset()
# calmar_obs = calmar_env.reset()
omega_obs = omega_env.reset()

profit_net_worths = [10000]
sortino_net_worths = [10000]
# calmar_net_worths = [10000]
omega_net_worths = [10000]

done = False
while not done:
    profit_action, profit_states = profit_model.predict(profit_obs)
    sortino_action, sortino_states = sortino_model.predict(sortino_obs)
    # calmar_action, calmar_states = calmar_model.predict(calmar_obs)
    omega_action, omega_states = omega_model.predict(omega_obs)

    profit_obs, profit_reward, done, info = profit_env.step(profit_action)
    sortino_obs, sortino_reward, done, info = sortino_env.step(sortino_action)
    # calmar_obs, calmar_reward, done, info = calmar_env.step(calmar_action)
    omega_obs, omega_reward, done, info = omega_env.step(omega_action)

    profit_net_worths.append(profit_net_worths[-1] + profit_reward[0])
    sortino_net_worths.append(sortino_net_worths[-1] + sortino_reward[0])
    # calmar_net_worths.append(calmar_net_worths[-1] + calmar_reward[0])
    omega_net_worths.append(omega_net_worths[-1] + omega_reward[0])

with open('./research/results/profit_net_worths_4.pkl', 'wb') as handle:
    pickle.dump(profit_net_worths, handle)

with open('./research/results/sortino_net_worths_4.pkl', 'wb') as handle:
    pickle.dump(sortino_net_worths, handle)

# with open('./research/results/calmar_net_worths_4.pkl', 'wb') as handle:
#     pickle.dump(calmar_net_worths, handle)

with open('./research/results/omega_net_worths_4.pkl', 'wb') as handle:
    pickle.dump(omega_net_worths, handle)


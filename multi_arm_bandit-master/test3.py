import numpy as np
import matplotlib.pyplot as plt
import math


class UCB:
    def __init__(self, c, actions):
        self.average_rewards = np.repeat(0., len(actions))     # 各腕の平均報酬
        self.ucbs = np.repeat(10., len(actions))   # UCB値
        self.counters = np.repeat(0, len(actions))  # 各腕の試行回数
        self.c = c
        self.all_conter = 0     # 全試行回数

    def select_action(self):
        # print(self.ucbs)
        action_id = np.argmax(self.ucbs)
        self.counters[action_id] += 1
        self.all_conter += 1
        return action_id

    def update_ucbs(self, action_id, reward):
        self.update_average_rewards(action_id, reward)
        self.ucbs[action_id] = self.average_rewards[action_id] + self.c * math.log(2.*self.all_conter)/np.sqrt(self.counters[action_id])

    def update_average_rewards(self, action_id, reward):
        self.average_rewards[action_id] = self.average_rewards[action_id] + (reward-self.average_rewards[action_id])/(self.counters[action_id]+1)


class SimpleRLAgent:
    def __init__(self, policy, actions):
        self.policy = policy
        self.actions = actions
        self.recent_action_id = None

    def act(self):
        action_id = self.policy.select_action()    # 行動選択
        # print(action_id)
        self.recent_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe(self, reward):
        self.policy.update_ucbs(self.recent_action_id, reward)


class Arm:
    def __init__(self, idx, mu, sd):
        self.idx = idx
        self.mu = mu
        self.sd = sd

    def pull(self):
        return np.random.normal(self.mu, self.sd)


class MultiArmBandit:

    def __init__(self, arm_confs):
        self.arms = self._init_arms(arm_confs)

    def _init_arms(self, arm_confs):
        arms = []
        for arm_conf in arm_confs:
            print(arm_conf)
            arm = Arm(arm_conf["id"], arm_conf["mu"], arm_conf["sd"])
            print(arm)
            arms.append(arm)

        return arms

    def step(self, arm_id):
        return self.arms[arm_id].pull()


if __name__ == '__main__':
    arm_confs = [{"id": 0, "mu": .1, "sd": .1},    # 平均 0.1、分散0.1の正規分布に従う乱数によって報酬を設定
                 {"id": 1, "mu": .5, "sd": .1},
                 {"id": 2, "mu": 2., "sd": .1},
                 {"id": 3, "mu": .2, "sd": .1},
                 {"id": 4, "mu": .4, "sd": .1}]

    game = MultiArmBandit(arm_confs=arm_confs)  # 5本のアームを設定
    policy = UCB(c=.2, actions=np.arange(len(arm_confs)))    # UCBアルゴリズム
    agent = SimpleRLAgent(policy=policy, actions=np.arange(len(arm_confs)))  # agentの設定
    nb_step = 100   # ステップ数
    reward_history = []
    count_selected_arm = {}
    for step in range(nb_step):
        arm_id = agent.act()    # レバーの選択
        reward = game.step(arm_id)  # レバーを引く
        agent.observe(reward) #　エージェントは報酬を受け取り学習
        aim_id = 7
        if arm_id not in count_selected_arm.keys():
           count_selected_arm[arm_id] = 0
        count_selected_arm[arm_id] += 1
        reward_history.append(reward)

    print(count_selected_arm)

    plt.plot(np.arange(nb_step), reward_history)
    plt.ylabel("reward")
    plt.xlabel("steps")
    plt.savefig("result_ucb.png")
    plt.show()
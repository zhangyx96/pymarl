import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer_new(nn.Module):
    def __init__(self, args):
        super(QMixer_new, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_action_dim = int(np.prod(args.state_shape)) + int(self.n_agents) #加入action信息
        self.state_dim = int(np.prod(args.state_shape)) #120
        self.embed_dim = args.mixing_embed_dim  #32

        self.hyper_w_1 = nn.Linear(self.state_action_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_action_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_action_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_action_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        actions = actions.squeeze() #去掉最后一维
        actions = actions.float()  #long 转 float
        states_actions = th.cat((states,actions),dim=2)
        states = states.reshape(-1, self.state_dim)
        states_actions = states_actions.reshape(-1, self.state_action_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states_actions))
        b1 = self.hyper_b_1(states_actions)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states_actions))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states_actions).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

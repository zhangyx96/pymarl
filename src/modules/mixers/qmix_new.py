import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#exp 6
class QMixer_new(nn.Module):
    def __init__(self, args):
        super(QMixer_new, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))   #120
        self.action_dim = int(args.n_actions*args.n_agents)

        self.embed_dim = args.mixing_embed_dim
        self.embed_dim_2 = 16

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim_2)

        #add the action layer
        self.hyper_w_2 = nn.Linear(self.action_dim,self.embed_dim * self.embed_dim_2)
        self.hyper_b_2 = nn.Linear(self.action_dim,self.embed_dim_2)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)  
        actions = actions.reshape(states.shape[0],-1) 
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer(action)
        w2 = th.abs(self.hyper_w_2(actions))
        b2 = self.hyper_b_2(actions)
        w2 = w2.view(-1,self.embed_dim,self.embed_dim_2)
        b2 = b2.view(-1, 1, self.embed_dim_2)
        hidden2 = F.elu(th.bmm(hidden, w2) + b2)

        # Third layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim_2, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden2, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

'''
#exp 6
class QMixer_new(nn.Module):
    def __init__(self, args):
        super(QMixer_new, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))   #120
        self.action_dim = int(args.n_actions*args.n_agents)

        self.embed_dim = args.mixing_embed_dim
        self.embed_dim_2 = 16

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim_2)

        #add the action layer
        self.hyper_w_2 = nn.Linear(self.action_dim,self.embed_dim * self.embed_dim_2)
        self.hyper_b_2 = nn.Linear(self.action_dim,self.embed_dim_2)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)  
        actions = actions.reshape(states.shape[0],-1) 
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer(action)
        w2 = th.abs(self.hyper_w_2(actions))
        b2 = self.hyper_b_2(actions)
        w2 = w2.view(-1,self.embed_dim,self.embed_dim_2)
        b2 = b2.view(-1, 1, self.embed_dim_2)
        hidden2 = F.elu(th.bmm(hidden, w2) + b2)

        # Third layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim_2, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden2, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

#exp2
class QMixer_new(nn.Module):
    def __init__(self, args):
        super(QMixer_new, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_action_dim = int(np.prod(args.state_shape)) + int(args.n_actions*self.n_agents) #加入action信息
        self.embed_dim = args.mixing_embed_dim  #32

        self.hyper_w_1 = nn.Linear(self.state_action_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_action_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_action_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_action_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions_onehot):
        bs = agent_qs.size(0)
        #actions = actions.squeeze() #去掉最后一维
        actions_onehot = actions_onehot.view(actions_onehot.size()[0],actions_onehot.size()[1],-1)
        #print(actions_onehot.size())
        actions = actions_onehot.float()  #long 转 float
        states_actions = th.cat((states,actions),dim=2)
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

# exp3
class QMixer_new(nn.Module):
    def __init__(self, args):
        super(QMixer_new, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))   #120
        self.action_dim = int(args.n_actions*args.n_agents)

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.action_dim, self.embed_dim)

        #self.hyper_w_3 = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.action_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        #print(agent_qs.size())
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)  
        actions = actions.reshape(states.shape[0],-1) 
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(actions))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(actions).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

# exp4
class QMixer_new(nn.Module):
    def __init__(self, args):
        super(QMixer_new, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))   #120
        self.action_dim = int(args.n_actions*args.n_agents)

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.action_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.action_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)  
        actions = actions.reshape(states.shape[0],-1) 
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(actions))
        b1 = self.hyper_b_1(actions)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

# exp5
class QMixer_new(nn.Module):
    def __init__(self, args):
        super(QMixer_new, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape)) 
        self.action_dim = int(args.n_actions*args.n_agents)

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.action_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.action_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.action_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.action_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)  
        actions = actions.reshape(states.shape[0],-1) 
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(actions))
        b1 = self.hyper_b_1(actions)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(actions))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(actions).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
'''
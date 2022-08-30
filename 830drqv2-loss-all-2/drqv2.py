# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import ipdb

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)



class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out
        
class STN(nn.Module):
    def __init__(self, input_size, output_size=6, linear_size=32,
                 num_stage=2, p_dropout=0.5, alpha_agnet=None):
        super(STN, self).__init__()
        # print('point 0')
        self.linear_size = linear_size
        print('linear_size: {}'.format(linear_size))
        self.p_dropout = p_dropout
        print('p_dropout: {}'.format(p_dropout))
        self.num_stage = num_stage
        print('num_stage: {}'.format(num_stage))

        # noise dim
        self.input_size = input_size
        print('theta generator input dim: {}'.format(self.input_size))
        # theta dim
        self.output_size = output_size
        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        # Initialize the weights/bias with identity transformation
        self.w2.weight.data.zero_()
        self.w2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.id_map = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float).cuda()
        self.id_map = torch.tensor([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], dtype=torch.float32).cuda()
        self.id_map_2 = torch.tensor([[[1, 0, 0],
                                    [0, 1, 0]]], dtype=torch.float32).cuda()
        self.pad_row = torch.tensor([[0., 0., 1.]], dtype=torch.float32).cuda()
        self.mse_loss = nn.MSELoss()

        self.alpha_agnet = alpha_agnet


    def div_loss(self, theta):
        id_maps = self.id_map_2.repeat(theta.size(0), 1, 1)
        # print('id_maps shape: {}'.format(id_maps.size()))
        # print('theta shape: {}'.format(theta.size()))
        # exit()
        return self.mse_loss(theta, id_maps)

    def inv_theta(self, theta):
        pad_rows = self.pad_row.repeat(theta.size(0), 1, 1)
        theta_padded = torch.cat([theta, pad_rows], dim=1)
        theta_padded_inv = torch.inverse(theta_padded)
        theta_inv = theta_padded_inv[:, 0:-1, :]
        return theta_inv

    def tf_func(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x_tf = F.grid_sample(x, grid)
        return x_tf

    def diversity_loss(self, input1, output1, input2, output2, eps):
        output_diff = F.mse_loss(output1, output2, reduction='none')
        assert output_diff.size() == output1.size()
        output_diff_vec = output_diff.view(output_diff.size(0), -1).mean(dim=1)
        assert len(output_diff_vec.size()) == 1
        # noise_diff_vec = F.l1_loss(noise, noise_2, reduction='none').sum(dim=1)
        input_diff_vec = F.mse_loss(input1, input2, reduction='none').mean(dim=1)
        assert len(input_diff_vec.size()) == 1
        loss = output_diff_vec / (input_diff_vec + eps)
        # loss = torch.clamp(loss, max=1)
        return loss.mean()

    def theta_diversity_loss(self, noise, theta, eps=1e-3):
        noise_2 = torch.randn_like(noise)
        theta_2 = self.localization(noise_2)
        loss = self.diversity_loss(noise, theta, noise_2, theta_2, eps)
        return loss

    def img_diversity_loss(self, x, x_tf, noise, eps=1e-1):
        noise_2 = torch.randn_like(noise)
        theta_2 = self.localization(noise_2)
        x_tf_2 = self.tf_func(x, theta_2)
        # version 2
        loss = self.diversity_loss(noise, x_tf, noise_2, x_tf_2, eps)
        return loss

    def localization(self, noise):
        # pre-processing
        y = self.w1(noise)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        theta = self.w2(y)
        theta = theta.view(-1, 2, 3)

        return theta

    def forward(self, noise, x, label, require_loss=False):
        theta = self.localization(noise)
        assert theta.size(0) == x.size(0)
        theta_inv = self.inv_theta(theta)
        assert theta_inv.size() == theta.size()
        # ipdb.set_trace()
        x_tf = self.tf_func(x, theta)
        x_tf_inv = self.tf_func(x, theta_inv)

        # get the transformed x and its corresponding label
        x_comb = torch.cat([x_tf, x_tf_inv], dim=0)
        label_comb = torch.cat([label, label], dim=0)
        ##change point 1
        shape_t = theta.size()
        n = shape_t[0]
        d_theta = []
        for i in range(n):
            d_theta.append(theta[i,:,:].trace())
        d_tensor = torch.tensor(d_theta)
        d_target = torch.ones(d_tensor.size())*self.alpha_agnet
        # ipdb.set_trace()
        # print(self.alpha_agnet)
        loss_theta = self.mse_loss(d_tensor, d_target)
         ##change point 1

        if not require_loss:
            return x_comb, label_comb
        else:
            x_tf_recon = self.tf_func(x_tf, theta_inv)
            # reconstruct x from inverse theta tf
            x_tf_inv_recon = self.tf_func(x_tf_inv, theta)
            return x_comb, label_comb, \
                   self.mse_loss(x, x_tf_recon)+self.mse_loss(x, x_tf_inv_recon), \
                   self.img_diversity_loss(x, x_tf, noise), loss_theta
# self.theta_cosine_diversity_loss(noise, theta)
# self.theta_diversity_loss(noise, theta)

##############



class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, alpha_agnet):


        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)




        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        #   stn layer settings
        aug_net_lr = 1e-5
        adam_beta1 = 0.5
        aug_net_weight_decay = 1e-3
        self.adv_weight_stn = 0.1

        # import ipdb.set_trace()

        self.aug_net = STN(input_size=1, linear_size=10, alpha_agnet=alpha_agnet).to(device)
        self.noise_dim = 1
        self.div_loss_weight = 0.1
        self.diversity_loss_weight = 0.1
        self.aug_net_optim = torch.optim.Adam(self.aug_net.parameters(),
                                              lr=aug_net_lr, betas=(adam_beta1, 0.999),
                                              weight_decay=aug_net_weight_decay)
        #   stn layer settings
        

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        #stn setting
        self.aug_net.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs_unencoder, obs_aug, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        
        # inner_num = 1
        # for j in range(inner_num):
        #     #stn training
        #     # ipdb.set_trace()
        #     noise = torch.randn(obs.size(0), self.noise_dim).to(self.device)
        #     input_aug_obs, target_augQ, div_loss, diversity_loss = \
        #            self.aug_net(noise, obs, target_Q, require_loss=True)
        #     action_comb = torch.cat([action, action], dim=0)
        #     input_aug_obs.register_hook(lambda grad: grad * (-self.adv_weight_stn))
        #     input_aug_obs = self.encoder(input_aug_obs)
        #     stn_target_Q1, stn_target_Q2 = self.critic(input_aug_obs, action_comb)
        #     stn_target_loss = F.mse_loss(stn_target_Q1, target_augQ)+F.mse_loss(stn_target_Q2, target_augQ)
        #     stn_loss = stn_target_loss + self.div_loss_weight * div_loss - self.diversity_loss_weight * diversity_loss
        #     stn_loss = stn_loss.mean()
        #     #logger.log('train_critic/stn_loss', stn_loss, step)
        #     #logger.log('train_critic/div_loss', div_loss, step)
        #     #logger.log('train_critic/diversity_loss', diversity_loss, step)
        #     self.aug_net_optim.zero_grad()
        #     stn_loss.backward()
        #     self.aug_net_optim.step()
        #     self.critic_opt.step()
        #     ##stn training

        #stn training
        noise = torch.randn(obs_unencoder.size(0), self.noise_dim).to(self.device)
        input_aug_obs, target_augQ, div_loss, diversity_loss, theta_loss = \
               self.aug_net(noise, obs_unencoder, target_Q, require_loss=True)
        action_comb = torch.cat([action, action], dim=0)
        # input_aug_obs.register_hook(lambda grad: grad * (-self.adv_weight_stn))
        input_aug_obs = self.encoder(input_aug_obs)
        stn_target_Q1, stn_target_Q2 = self.critic_target(input_aug_obs, action_comb)
        stn_target_loss = F.mse_loss(stn_target_Q1, target_augQ)+F.mse_loss(stn_target_Q2, target_augQ)
        stn_loss = stn_target_loss + self.div_loss_weight * div_loss - self.diversity_loss_weight * diversity_loss + theta_loss
        stn_loss = stn_loss.mean()
        #logger.log('train_critic/stn_loss', stn_loss, step)
        #logger.log('train_critic/div_loss', div_loss, step)
        #logger.log('train_critic/diversity_loss', diversity_loss, step)
        self.aug_net_optim.zero_grad()
        # self.encoder_opt.zero_grad()
        self.critic_opt.zero_grad()
        stn_loss.backward()
        self.aug_net_optim.step()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.)
        # nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.)
        self.critic_opt.step()
        # self.encoder_opt.step()
        ##stn training


        #stn change
        #Q1, Q2 = self.critic(obs, action)
        self.encoder_opt.zero_grad()
        self.critic_opt.zero_grad()
        Q1, Q2 = self.critic(obs_aug, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        # self.encoder_opt.zero_grad(set_to_none=True)
        # self.critic_opt.zero_grad(set_to_none=True)  适用于torch1.10代码，已改
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        # self.actor_opt.zero_grad(set_to_none=True) 
        self.actor_opt.zero_grad() 
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment

        #stn change1
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        # obs = self.encoder(obs)
        #stn change
        obs_encoder = self.encoder(obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        #stn change
        metrics.update(
            self.update_critic(obs.float(), obs_encoder, action, reward, discount, next_obs, step))

        # update actor
        # metrics.update(self.update_actor(obs.detach(), step))
        #stn change
        metrics.update(self.update_actor(obs_encoder.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

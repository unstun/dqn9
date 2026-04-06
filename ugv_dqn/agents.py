"""用于 UGV 路径规划的 DQN 系列强化学习智能体。

提供
----
- AgentConfig          所有超参数的冻结数据类（gamma、lr、eps、DQfD margin 等）
- parse_rl_algo()      规范名称解析："mlp-dqn" / "cnn-ddqn" / 旧版别名
- DQNFamilyAgent       统一智能体，支持 DQN、Double-DQN (DDQN) 和 Polyak-DDQN (PDDQN)，
                       可选 MLP 或 CNN Q 网络、n-step 回报、动作掩码，以及
                       DQfD 风格的专家 margin + 行为克隆损失。

数据流
------
env.step() -> agent.observe() -> replay_buffer -> agent.update() -> Q 网络梯度
                                                   ^ 可选 DQfD 演示损失
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ugv_dqn.networks import CNNQNetwork, MLPQNetwork, infer_flat_obs_cnn_layout
from ugv_dqn.replay_buffer import ReplayBuffer
from ugv_dqn.schedules import linear_epsilon


@dataclass(frozen=True)
class AgentConfig:
    # Forest 环境（dt=0.05s，长时间跨度）需要更高的折扣因子；gamma=0.9 会使
    # 终端奖励在几十步后基本消失。
    gamma: float = 0.995
    n_step: int = 1
    learning_rate: float = 5e-4
    replay_capacity: int = 100_000
    batch_size: int = 128
    target_update_steps: int = 1000
    # 当 >0 时，每个训练步使用 Polyak（软）目标更新：
    #   target = (1 - tau) * target + tau * online
    # 当 =0 时，每隔 `target_update_steps` 步进行硬更新。
    target_update_tau: float = 0.0
    grad_clip_norm: float = 10.0

    eps_start: float = 0.9
    eps_final: float = 0.01
    eps_decay: int = 2000

    hidden_layers: int = 3
    hidden_dim: int = 256

    # Dueling DQN (Wang et al., 2016)：将 Q 拆分为 V(s) + A(s,a) - mean(A)。
    dueling: bool = False
    # 在 CNN 特征图上使用空间多头自注意力。
    mha: bool = False
    mha_heads: int = 4

    # 消融模块（仅 CNN）
    coord_attn: bool = False    # Coordinate Attention (Hou et al., CVPR 2021)
    noisy: bool = False         # NoisyNet (Fortunato et al., ICLR 2018)
    noisy_reset_interval: int = 4  # 每 N 个更新步重新采样噪声
    fadc: bool = False          # 频率自适应膨胀卷积 (Chen et al., CVPR 2024)
    deform: bool = False        # 可变形卷积 v2（通过 torchvision）
    iqn: bool = False           # Implicit Quantile Networks (Dabney et al., ICML 2018)
    iqn_cos: int = 64           # IQN 余弦嵌入维度
    iqn_quantiles: int = 8      # IQN 分位数采样数量

    # Munchausen DQN (Vieillard et al., NeurIPS 2020)
    munchausen: bool = False    # 将 scaled log-policy 加到即时奖励上
    m_alpha: float = 0.9        # Munchausen 缩放系数
    m_tau: float = 0.03         # 策略 softmax 温度
    m_lo: float = -1.0          # log-policy 裁剪下界

    # 专家 margin 损失（DQfD 风格），用于 forest 环境稳定训练。
    demo_margin: float = 0.8
    demo_lambda: float = 1.0
    demo_ce_lambda: float = 1.0


AlgoArch = Literal["mlp", "cnn"]
AlgoBase = Literal["dqn", "ddqn"]


def parse_rl_algo(algo: str) -> tuple[str, AlgoArch, AlgoBase, bool]:
    """返回 (canonical_name, arch, base_algo, is_legacy_alias)。"""

    a = str(algo).lower().strip()
    if a in {"dqn", "ddqn"}:
        base: AlgoBase = "dqn" if a == "dqn" else "ddqn"
        return (f"mlp-{base}", "mlp", base, True)

    if a == "iddqn":
        # 旧版名称（避免与已发表的"IDDQN"冲突）。
        # 本项目中用于 Polyak/软目标 Double DQN 变体。
        return ("mlp-pddqn", "mlp", "ddqn", True)

    if a == "cnn-iddqn":
        # 旧版名称（避免与已发表的"IDDQN"冲突）。
        return ("cnn-pddqn", "cnn", "ddqn", True)

    supported = {"mlp-dqn", "mlp-ddqn", "mlp-pddqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn"}
    if a in supported:
        arch_s, variant = a.split("-", 1)
        arch: AlgoArch = "mlp" if arch_s == "mlp" else "cnn"
        base: AlgoBase = "dqn" if variant == "dqn" else "ddqn"
        return (a, arch, base, False)

    raise ValueError(
        "algo must be one of: mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn "
        "(legacy: dqn ddqn iddqn cnn-iddqn)"
    )


class DQNFamilyAgent:
    def __init__(
        self,
        algo: str,
        obs_dim: int,
        n_actions: int,
        *,
        config: AgentConfig,
        seed: int = 0,
        device: str | torch.device = "cpu",
        cnn_drop_edt: bool = False,
    ) -> None:
        canonical_algo, arch, base_algo, _legacy = parse_rl_algo(algo)
        self.algo = canonical_algo
        self.arch = arch
        self.base_algo = base_algo
        self.config = config
        self.device = torch.device(device)
        self.cnn_drop_edt = bool(cnn_drop_edt)

        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # DQN 是基线（普通 Q-learning）。
        # DDQN 保持相同架构，但使用 Double DQN TD 目标（在线网络选动作 + 目标网络评估）。
        #
        # 观测拆分：CNN 使用全部 3 个地图通道（occ + cost + EDT 间隙），
        # MLP 丢弃 EDT 通道（最后 N² 维），避免 MLP 无法利用的空间特征
        # 引入额外噪声。参见 repro_20260222_cnn_edt_channel.json。
        # 当 cnn_drop_edt=True 时，CNN 也丢弃 EDT 通道（消融实验）。
        self._env_obs_dim = int(obs_dim)
        effective_obs_dim = int(obs_dim)

        self._net_cls: type[nn.Module]
        self._net_kwargs: dict[str, object]
        if self.arch == "cnn":
            layout = infer_flat_obs_cnn_layout(int(obs_dim))
            _cnn_extra = {
                "dueling": bool(config.dueling),
                "mha": bool(config.mha),
                "mha_heads": int(config.mha_heads),
                "coord_attn": bool(config.coord_attn),
                "noisy": bool(config.noisy),
                "fadc": bool(config.fadc),
                "deform": bool(config.deform),
                "iqn": bool(config.iqn),
                "iqn_cos": int(config.iqn_cos),
                "iqn_quantiles": int(config.iqn_quantiles),
            }
            if self.cnn_drop_edt:
                # 消融实验：去除 EDT 通道，仅保留 occ + cost（2 个通道）
                effective_obs_dim = int(layout.scalar_dim) + 2 * int(layout.map_size) ** 2
                self._net_cls = CNNQNetwork
                self._net_kwargs = {
                    "scalar_dim": int(layout.scalar_dim),
                    "map_channels": 2,
                    "map_size": int(layout.map_size),
                    **_cnn_extra,
                }
            else:
                self._net_cls = CNNQNetwork
                self._net_kwargs = {
                    "scalar_dim": int(layout.scalar_dim),
                    "map_channels": int(layout.map_channels),
                    "map_size": int(layout.map_size),
                    **_cnn_extra,
                }
        else:
            # 当存在第 3 个地图通道（EDT）时去除它：11+3*N² → 11+2*N²。
            # scalar_only 模式下 obs_dim=11，无地图通道，无需裁剪。
            map_rem = int(obs_dim) - 11
            if map_rem > 0 and map_rem % 3 == 0:
                n_sq = map_rem // 3
                effective_obs_dim = 11 + 2 * n_sq
            self._net_cls = MLPQNetwork
            self._net_kwargs = {}

        obs_dim = effective_obs_dim
        self.q = self._net_cls(obs_dim, n_actions, hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers, **self._net_kwargs).to(self.device)
        self.q_target = self._net_cls(obs_dim, n_actions, hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers, **self._net_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.replay = ReplayBuffer(config.replay_capacity, obs_dim, n_actions, rng=self._rng)

        self._train_steps = 0
        self._noise_counter = 0  # 用于 NoisyNet 重置间隔
        self._n_actions = int(n_actions)
        self._obs_dim = int(obs_dim)
        self._n_step = int(max(1, int(getattr(config, "n_step", 1))))
        # (obs, action, reward, next_obs, done, demo, next_action_mask)
        self._nstep_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, bool, np.ndarray | None]] = deque()

    def _rebuild_networks(
        self,
        net_cls: type[nn.Module],
        *,
        hidden_dim: int,
        hidden_layers: int,
        net_kwargs: dict[str, object] | None = None,
    ) -> None:
        net_kwargs = {} if net_kwargs is None else dict(net_kwargs)
        self.q = net_cls(self._obs_dim, self._n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers, **net_kwargs).to(self.device)
        self.q_target = net_cls(self._obs_dim, self._n_actions, hidden_dim=hidden_dim, hidden_layers=hidden_layers, **net_kwargs).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.config.learning_rate)

    def epsilon(self, episode: int) -> float:
        return linear_epsilon(
            episode,
            eps_start=self.config.eps_start,
            eps_final=self.config.eps_final,
            decay_episodes=self.config.eps_decay,
        )

    def _prep_obs(self, obs: np.ndarray) -> np.ndarray:
        """将环境观测裁剪到当前智能体网络使用的有效维度。"""
        if self._obs_dim < self._env_obs_dim:
            return np.asarray(obs, dtype=np.float32).ravel()[: self._obs_dim]
        return np.asarray(obs, dtype=np.float32).ravel()

    def act(self, obs: np.ndarray, *, episode: int, explore: bool = True) -> int:
        # NoisyNet：不使用 epsilon-greedy；权重中的噪声提供探索
        if explore and not self.config.noisy and (self._rng.random() < self.epsilon(episode)):
            return int(self._rng.integers(0, self._n_actions))

        obs = self._prep_obs(obs)
        with torch.no_grad():
            x = torch.from_numpy(obs).to(self.device)
            q = self.q(x.unsqueeze(0)).squeeze(0)
            return int(torch.argmax(q).item())

    def act_masked(
        self,
        obs: np.ndarray,
        *,
        episode: int,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """带可选布尔动作掩码的 epsilon-greedy 动作选择。"""

        mask = None
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            if mask.size != self._n_actions:
                raise ValueError("action_mask must have shape (n_actions,)")

        # NoisyNet：不使用 epsilon-greedy；权重中的噪声提供探索
        if explore and not self.config.noisy and (self._rng.random() < self.epsilon(episode)):
            if mask is None:
                return int(self._rng.integers(0, self._n_actions))
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                return int(self._rng.integers(0, self._n_actions))
            return int(self._rng.choice(idxs))

        obs = self._prep_obs(obs)
        with torch.no_grad():
            x = torch.from_numpy(obs).to(self.device)
            q = self.q(x.unsqueeze(0)).squeeze(0)
            if mask is not None:
                q = q.clone()
                q[torch.from_numpy(~mask).to(self.device)] = torch.finfo(q.dtype).min
            return int(torch.argmax(q).item())

    def top_actions(self, obs: np.ndarray, *, k: int) -> np.ndarray:
        """返回按 Q 值降序排列的前 k 个动作索引。"""
        obs = self._prep_obs(obs)
        kk = int(max(1, int(k)))
        with torch.no_grad():
            x = torch.from_numpy(obs).to(self.device)
            q = self.q(x.unsqueeze(0)).squeeze(0)
            kk = int(min(int(kk), int(q.numel())))
            return torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()

    def _add_to_replay(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        next_action_mask: np.ndarray | None,
        demo: bool,
        n_steps: int,
    ) -> None:
        self.replay.add(
            obs,
            int(action),
            float(reward),
            next_obs,
            bool(done),
            next_action_mask=next_action_mask,
            demo=bool(demo),
            n_steps=int(n_steps),
        )

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        *,
        demo: bool = False,
        truncated: bool = False,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        """记录一次转移（支持 n-step 回报）。

        `done` 应反映 *真正的* 终止状态（碰撞/到达目标）。对于时间限制导致的
        回合结束请使用 `truncated=True`，这样 n-step 缓冲区不会跨回合泄漏，
        同时仍允许从最终状态进行自举。
        """
        obs = self._prep_obs(obs)
        next_obs = self._prep_obs(next_obs)

        if int(self._n_step) <= 1:
            self._add_to_replay(
                obs,
                int(action),
                float(reward),
                next_obs,
                bool(done),
                next_action_mask=next_action_mask,
                demo=bool(demo),
                n_steps=1,
            )
            if bool(done) or bool(truncated):
                self.end_episode()
            return

        self._nstep_buffer.append((obs, int(action), float(reward), next_obs, bool(done), bool(demo), next_action_mask))

        episode_end = bool(done) or bool(truncated)
        if (len(self._nstep_buffer) < int(self._n_step)) and not episode_end:
            return

        self._flush_nstep_buffer(force=episode_end)

    def _flush_nstep_buffer(self, *, force: bool) -> None:
        if int(self._n_step) <= 1:
            return

        gamma = float(self.config.gamma)
        n_target = int(self._n_step)

        # 当 `force` 为 True（回合边界）时，以截断的 n-step 视野刷新所有缓冲。
        while self._nstep_buffer:
            horizon = min(int(n_target), int(len(self._nstep_buffer)))
            ret = 0.0
            n_used = 0
            next_obs_n = self._nstep_buffer[0][3]
            next_mask_n = self._nstep_buffer[0][6]
            done_n = False
            for i in range(horizon):
                _o, _a, r, no, d, _demo, nm = self._nstep_buffer[i]
                ret += (gamma**i) * float(r)
                n_used = i + 1
                next_obs_n = no
                next_mask_n = nm
                done_n = bool(d)
                if done_n:
                    break

            obs0, a0, _r0, _no0, _d0, demo0, _nm0 = self._nstep_buffer[0]
            self._add_to_replay(
                obs0,
                int(a0),
                float(ret),
                next_obs_n,
                bool(done_n),
                next_action_mask=next_mask_n,
                demo=bool(demo0),
                n_steps=int(n_used),
            )
            self._nstep_buffer.popleft()

            # 如果不在回合边界，每步只发出一个转移。
            if not bool(force):
                break

    def end_episode(self) -> None:
        """在回合边界刷新所有挂起的 n-step 转移。"""
        if int(self._n_step) <= 1:
            return
        self._flush_nstep_buffer(force=True)

    def pretrain_on_demos(self, *, steps: int) -> int:
        """在演示转移上进行有监督预热（DQfD 风格的稳定器）。"""

        n_steps = int(steps)
        if n_steps <= 0:
            return 0
        if len(self.replay) < int(self.config.batch_size):
            return 0

        demo_lambda = float(getattr(self.config, "demo_lambda", 0.0))
        demo_margin = float(getattr(self.config, "demo_margin", 0.0))
        demo_ce_lambda = float(getattr(self.config, "demo_ce_lambda", 0.0))

        trained = 0
        for _ in range(n_steps):
            if len(self.replay) < int(self.config.batch_size):
                break

            batch = self.replay.sample(self.config.batch_size)
            obs = torch.from_numpy(batch.obs).to(self.device)
            actions = torch.from_numpy(batch.actions).to(self.device)
            demos = torch.from_numpy(batch.demos).to(self.device)

            demo_mask = demos.float().clamp(0.0, 1.0) > 0.0
            if not bool(torch.any(demo_mask)):
                continue

            q_all = self.q(obs)
            q_demo = q_all[demo_mask]
            a_demo = actions.long()[demo_mask]

            loss = torch.tensor(0.0, device=self.device)
            if demo_ce_lambda > 0.0:
                loss = loss + float(demo_ce_lambda) * F.cross_entropy(q_demo, a_demo, reduction="mean")

            if demo_lambda > 0.0 and demo_margin > 0.0:
                q_a = q_demo.gather(1, a_demo.view(-1, 1)).squeeze(1)
                q_other = q_demo.clone()
                q_other.scatter_(1, a_demo.view(-1, 1), torch.finfo(q_other.dtype).min)
                q_max_other = q_other.max(dim=1).values
                margin = torch.relu(q_max_other + float(demo_margin) - q_a).mean()
                loss = loss + float(demo_lambda) * margin

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.config.grad_clip_norm)
            self.optimizer.step()
            trained += 1

        if trained > 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return int(trained)

    def update(self, *, rew_normalizer: object | None = None) -> dict[str, float]:
        if len(self.replay) < self.config.batch_size:
            return {}

        # NoisyNet：每 N 个更新步重新采样噪声（默认 4）
        if self.config.noisy:
            self._noise_counter += 1
            if self._noise_counter >= self.config.noisy_reset_interval:
                self._noise_counter = 0
                if hasattr(self.q, "reset_noise"):
                    self.q.reset_noise()
                if hasattr(self.q_target, "reset_noise"):
                    self.q_target.reset_noise()

        batch = self.replay.sample(self.config.batch_size)
        obs = torch.from_numpy(batch.obs).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        # V9：在采样时使用当前运行统计量归一化奖励。
        if rew_normalizer is not None:
            _norm_t = getattr(rew_normalizer, "normalize_tensor", None)
            if callable(_norm_t):
                rewards = _norm_t(rewards)
        next_obs = torch.from_numpy(batch.next_obs).to(self.device)
        next_action_masks = torch.from_numpy(batch.next_action_masks).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        n_steps = torch.from_numpy(batch.n_steps).to(self.device)
        demos = torch.from_numpy(batch.demos).to(self.device)

        if self.config.iqn and hasattr(self.q, "forward_quantiles"):
            # ----- IQN 分位数 Huber 损失 (Dabney et al., ICML 2018) -----
            K = int(self.config.iqn_quantiles)
            B = obs.shape[0]

            # 在线网络：采样 τ_i，获取分位数 Q 值
            tau_i = torch.rand(B, K, device=self.device)
            z_theta = self.q.forward_quantiles(obs, tau_i)              # (B, K, n_actions)
            q_all = z_theta.mean(dim=1)                                 # (B, n_actions)
            q_values = q_all.gather(1, actions.view(-1, 1)).squeeze(1)
            z_a = z_theta.gather(2, actions.view(-1, 1, 1).expand(-1, K, 1)).squeeze(2)  # (B, K)

            with torch.no_grad():
                mask = next_action_masks.to(torch.bool)
                # 动作选择（Double DQN 风格：在线网络选动作）
                if self.base_algo == "ddqn":
                    q_next_online = self.q(next_obs)
                    q_next_online = q_next_online.masked_fill(~mask, torch.finfo(q_next_online.dtype).min)
                    next_actions_iqn = torch.argmax(q_next_online, dim=1)
                else:
                    q_next_target_mean = self.q_target(next_obs)
                    q_next_target_mean = q_next_target_mean.masked_fill(~mask, torch.finfo(q_next_target_mean.dtype).min)
                    next_actions_iqn = torch.argmax(q_next_target_mean, dim=1)

                # 目标网络：采样 τ_j，获取目标分位数
                tau_j = torch.rand(B, K, device=self.device)
                z_target = self.q_target.forward_quantiles(next_obs, tau_j)  # (B, K, n_actions)
                z_next = z_target.gather(2, next_actions_iqn.view(-1, 1, 1).expand(-1, K, 1)).squeeze(2)  # (B, K)
                z_next = torch.where(torch.isfinite(z_next), z_next, torch.zeros_like(z_next))

                gamma = float(self.config.gamma)
                gamma_n = torch.pow(torch.tensor(gamma, device=self.device, dtype=torch.float32), n_steps.float())
                T_z = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * (gamma_n.unsqueeze(1) * z_next)  # (B, K)

            # ρ_τ^κ(δ) = |τ - I{δ<0}| · L_κ(δ) / κ,  κ=1
            delta = T_z.unsqueeze(1) - z_a.unsqueeze(2)             # (B, K_i, K_j)
            huber = torch.where(delta.abs() <= 1.0, 0.5 * delta.pow(2), delta.abs() - 0.5)
            rho = (tau_i.unsqueeze(2) - (delta < 0).float()).abs() * huber
            td_loss = rho.sum(dim=2).mean(dim=1).mean()
        else:
            # ----- 标准 TD 损失路径 -----
            q_all = self.q(obs)
            q_values = q_all.gather(1, actions.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                mask = next_action_masks.to(torch.bool)
                if self.base_algo == "ddqn":
                    q_next_online = self.q(next_obs)
                    q_next_online = q_next_online.masked_fill(~mask, torch.finfo(q_next_online.dtype).min)
                    next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)

                    q_next_target = self.q_target(next_obs)
                    q_next_target = q_next_target.masked_fill(~mask, torch.finfo(q_next_target.dtype).min)
                    next_q = q_next_target.gather(1, next_actions).squeeze(1)
                else:
                    q_next_target = self.q_target(next_obs)
                    q_next_target = q_next_target.masked_fill(~mask, torch.finfo(q_next_target.dtype).min)
                    next_q = q_next_target.max(dim=1).values

                next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
                gamma = float(self.config.gamma)
                gamma_n = torch.pow(torch.tensor(gamma, device=self.device, dtype=torch.float32), n_steps.to(torch.float32))

                if self.config.munchausen:
                    # ----- Munchausen DQN (Vieillard et al., NeurIPS 2020) -----
                    m_tau = float(self.config.m_tau)
                    m_alpha = float(self.config.m_alpha)
                    m_lo = float(self.config.m_lo)

                    q_tgt_curr = self.q_target(obs)
                    v_curr = q_tgt_curr.max(dim=1, keepdim=True)[0]
                    logsum_curr = torch.logsumexp((q_tgt_curr - v_curr) / m_tau, dim=1, keepdim=True)
                    tau_log_pi_curr = q_tgt_curr - v_curr - m_tau * logsum_curr
                    tau_log_pi_a = tau_log_pi_curr.gather(1, actions.view(-1, 1)).squeeze(1)
                    tau_log_pi_a = tau_log_pi_a.clamp(m_lo, 0.0)

                    q_tgt_next_raw = self.q_target(next_obs)
                    q_tgt_next_m = q_tgt_next_raw.masked_fill(~mask, torch.finfo(q_tgt_next_raw.dtype).min)
                    v_next = q_tgt_next_m.max(dim=1, keepdim=True)[0]
                    logsum_next = torch.logsumexp((q_tgt_next_m - v_next) / m_tau, dim=1, keepdim=True)
                    tau_log_pi_next = q_tgt_next_m - v_next - m_tau * logsum_next
                    pi_next = torch.softmax(q_tgt_next_m / m_tau, dim=1)
                    soft_v_next = (pi_next * (q_tgt_next_m - tau_log_pi_next)).sum(dim=1)
                    soft_v_next = torch.where(torch.isfinite(soft_v_next), soft_v_next, torch.zeros_like(soft_v_next))

                    target = (rewards + m_alpha * tau_log_pi_a) + (1.0 - dones) * (gamma_n * soft_v_next)
                else:
                    target = rewards + (1.0 - dones) * (gamma_n * next_q)

            losses = self.loss_fn(q_values, target)
            td_loss = losses.mean()

        # 专家大 margin 损失（DQfD）。仅应用于 `demo` 转移。
        demo_lambda = float(getattr(self.config, "demo_lambda", 0.0))
        demo_margin = float(getattr(self.config, "demo_margin", 0.0))
        margin_loss = torch.tensor(0.0, device=self.device)
        if demo_lambda > 0.0 and demo_margin > 0.0:
            demo_mask = demos.float().clamp(0.0, 1.0)
            if torch.any(demo_mask > 0.0):
                q_other = q_all.clone()
                q_other.scatter_(1, actions.view(-1, 1), torch.finfo(q_other.dtype).min)
                q_max_other = q_other.max(dim=1).values
                margin = torch.relu(q_max_other + float(demo_margin) - q_values)
                margin_loss = (margin * demo_mask).mean()

        # 演示转移上的专家行为克隆损失（在静态地图上起到强稳定器作用）。
        demo_ce_lambda = float(getattr(self.config, "demo_ce_lambda", 0.0))
        ce_loss = torch.tensor(0.0, device=self.device)
        if demo_ce_lambda > 0.0:
            demo_mask = demos.float().clamp(0.0, 1.0)
            if torch.any(demo_mask > 0.0):
                ce = F.cross_entropy(q_all, actions.long(), reduction="none")
                ce_loss = (ce * demo_mask).mean()

        loss = td_loss + float(demo_lambda) * margin_loss + float(demo_ce_lambda) * ce_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=self.config.grad_clip_norm)
        self.optimizer.step()

        self._train_steps += 1
        tau = float(getattr(self.config, "target_update_tau", 0.0))
        if tau > 0.0:
            with torch.no_grad():
                for p_t, p in zip(self.q_target.parameters(), self.q.parameters(), strict=False):
                    p_t.data.lerp_(p.data, float(tau))
        elif self._train_steps % self.config.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "margin_loss": float(margin_loss.item()),
            "ce_loss": float(ce_loss.item()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algo": self.algo,
            "arch": self.arch,
            "base_algo": self.base_algo,
            "network": self.arch,
            "network_kwargs": dict(self._net_kwargs),
            "obs_dim": int(self._obs_dim),
            "n_actions": int(self._n_actions),
            "config": self.config.__dict__,
            "q_state_dict": self.q.state_dict(),
            "q_target_state_dict": self.q_target.state_dict(),
            "train_steps": self._train_steps,
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        q_sd = payload["q_state_dict"]
        q_target_sd = payload.get("q_target_state_dict", {})

        payload_algo = payload.get("algo", self.algo)
        canonical_algo, arch, base_algo, _legacy = parse_rl_algo(str(payload_algo))
        self.algo = canonical_algo
        self.arch = arch
        self.base_algo = base_algo

        network = str(payload.get("network", self.arch)).lower().strip()
        if network in {"plain", "qnetwork", "mlp"}:
            net_cls: type[nn.Module] = MLPQNetwork
            net_kwargs: dict[str, object] = {}
            self.arch = "mlp"
        elif network == "cnn":
            net_cls = CNNQNetwork
            net_kwargs_raw = payload.get("network_kwargs") or {}
            if not isinstance(net_kwargs_raw, dict):
                net_kwargs_raw = {}
            if not net_kwargs_raw:
                layout = infer_flat_obs_cnn_layout(int(self._obs_dim))
                net_kwargs = {"scalar_dim": layout.scalar_dim, "map_channels": layout.map_channels, "map_size": layout.map_size}
            else:
                net_kwargs = {str(k): v for k, v in net_kwargs_raw.items()}
            self.arch = "cnn"
        else:
            raise ValueError(f"Unsupported network type in checkpoint: {network!r}")

        self._net_cls = net_cls
        self._net_kwargs = dict(net_kwargs)

        cfg = payload.get("config") or {}
        hidden_dim = int(cfg.get("hidden_dim", self.config.hidden_dim))
        hidden_layers = int(cfg.get("hidden_layers", self.config.hidden_layers))

        # 架构可能在不同实验间变化（hidden_dim/layers）。当形状不匹配时重建网络。
        try:
            self.q.load_state_dict(q_sd, strict=True)
        except RuntimeError:
            self._rebuild_networks(net_cls, hidden_dim=hidden_dim, hidden_layers=hidden_layers, net_kwargs=net_kwargs)

        self.q.load_state_dict(q_sd, strict=True)
        if q_target_sd:
            try:
                self.q_target.load_state_dict(q_target_sd, strict=True)
            except RuntimeError:
                # 如果检查点早于保存目标网络的版本（或形状不匹配），则回退到同步目标网络。
                self.q_target.load_state_dict(self.q.state_dict(), strict=True)
        else:
            self.q_target.load_state_dict(self.q.state_dict(), strict=True)
        self._train_steps = int(payload.get("train_steps", 0))

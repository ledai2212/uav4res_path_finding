"""QMIX + môi trường UAV phủ bản đồ -- gộp toàn bộ vào một file.

Chạy từ thư mục gốc dự án (vì qmix_model.pt, output.txt, ./images/uav.png là đường dẫn tương đối):
    python simulation-scenarios/qmix/qmix.py [--total-episodes 5000 --render False ...]
hoặc:
    python main.py qmix [--render False ...]

Bố cục file:
    1. Đường dẫn + import
    2. Args (cấu hình)
    3. EpisodeBuffer
    4. Mạng nơ-ron (AgentQNetwork, QMixer)
    5. Môi trường UAVSwarmCoverageEnv
    6. Render pygame
    7. Rollout + huấn luyện (trainer)
    8. Checkpoint
    9. Lưu best model (thay cho watcher_save.py)
   10. Hàm dựng env/mạng dùng chung (cả test_demo.py cũng dùng)
   11. main()
"""
# ============================================================================
# 1. ĐƯỜNG DẪN + IMPORT
# ============================================================================
import glob
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# Thêm thư mục gốc dự án (chứa input.py, Map.py, utils.py, ...) vào sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np                      # noqa: E402
import pygame                           # noqa: E402
import torch                            # noqa: E402
import torch.nn as nn                   # noqa: E402
import torch.nn.functional as F         # noqa: E402
import tyro                             # noqa: E402

from Drawer import Drawer               # noqa: E402
from Map import Map, Point              # noqa: E402
from Measurer import Measurer           # noqa: E402
from Swarm import Swarm                 # noqa: E402
from UAV import UAV                     # noqa: E402
from utils import (                     # noqa: E402
    find_path,                    # định tuyến về trạm sạc (use_energy)
    expand_path_to_grid_steps,    # bung đường Theta* thành từng bước lưới
    calculate_centroid_priority,  # tạo cluster (use_clusters)
)
from input import (                     # noqa: E402
    state, priority, uav_start,
    cell_size, cell_radius, maptype,
    uav_distance, time_charge, dis_threshold, base,
    min_speed, max_speed,
)


# ============================================================================
# 2. CẤU HÌNH
# ============================================================================
@dataclass
class Args:
    exp_name: str = "qmix"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    """bật để theo dõi bằng Weights & Biases"""
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None

    # --- environment ---
    num_uavs: int = 7
    obs_radius: int = 3
    """bán kính quan sát cục bộ; 3 -> cửa sổ 7x7"""
    spawn_jitter: bool = False
    """tản UAV ra các ô lân cận uav_start thay vì chồng hết lên 1 ô"""
    allow_corner_cutting: bool = True
    """cho phép đi chéo khi một trong hai ô bên cạnh là vật cản"""
    max_steps_per_episode: int = 1500
    step_penalty: float = 0.01
    invalid_penalty: float = 0.8
    priority_weight: float = 0.1
    """thưởng ô mới = 1 + priority_weight * priority"""
    use_energy: bool = True
    """mô hình pin: bay về trạm sạc khi sắp hết, thêm feature pin vào obs/state"""
    use_clusters: bool = True
    """phủ lần lượt từng cluster thay vì quét cả bản đồ một lượt"""
    use_measurer: bool = False
    """ghi thời gian/chi phí mỗi episode vào output.txt (type=qmix_v2)"""

    # --- reward shaping ---
    r_collision: float = -0.5
    r_cluster_direction: float = 7.0
    revisit_penalty: float = 0.2
    no_interest_penalty: float = 1
    shaping_coef: float = 1.5

    # --- QMIX / training ---
    total_episodes: int = 5000
    buffer_size: int = 150
    batch_size: int = 16
    gamma: float = 0.99
    learning_rate: float = 1e-4
    hidden_dim: int = 128
    mixing_embed_dim: int = 32
    grad_clip: float = 10.0
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.6
    learning_starts: int = 32
    train_every: int = 1
    tau: float = 0.005
    seq_len: int = 128
    target_update_interval: int = 200

    # --- checkpoint ---
    save_model: bool = True
    model_path: str = "qmix_model.pt"
    resume: bool = True
    save_every: int = 1

    # --- best model (thay cho watcher_save.py) ---
    save_best: bool = True
    """tự lưu best_*.pt khi episode đạt 100% coverage, ít bước và điểm cao hơn kỷ lục"""
    best_max_steps: int = 1350
    """chỉ lưu nếu số bước < giá trị này"""
    best_score_floor: float = 9200.0
    """điểm sàn ban đầu; sau đó kỷ lục = max(sàn, điểm cao nhất trong các best_*.pt đã có)"""

    # --- rendering ---
    render: bool = True
    render_every: int = 25
    render_fps: int = 60


# ============================================================================
# 3. REPLAY BUFFER (lưu nguyên episode cho GRU)
# ============================================================================
class EpisodeBuffer:
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.episodes = []

    def add(self, episode):
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def __len__(self):
        return len(self.episodes)

    def sample(self, batch_size):
        batch = random.sample(self.episodes, min(batch_size, len(self.episodes)))
        max_len = max(len(ep["reward"]) for ep in batch)
        return batch, max_len


# ============================================================================
# 4. MẠNG NƠ-RON
# ============================================================================
class AgentQNetwork(nn.Module):
    """CNN (3 kênh: state, priority, cluster) + GRU + đầu Q."""

    def __init__(self, obs_dim, n_actions, n_agents, hidden_dim=128, obs_window=7):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.obs_window = obs_window
        self.spatial_dim = obs_window * obs_window

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_size = 32 * (obs_window - 2) * (obs_window - 2)
        non_spatial_dim = obs_dim + n_agents - (3 * self.spatial_dim)

        self.fc1 = nn.Linear(cnn_out_size + non_spatial_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs_agentid, hidden_state):
        bs = obs_agentid.shape[0]
        spatial = obs_agentid[:, :3 * self.spatial_dim]
        non_spatial = obs_agentid[:, 3 * self.spatial_dim:]
        images = spatial.reshape(bs, 3, self.obs_window, self.obs_window)
        cnn_out = self.cnn(images)
        x = F.relu(self.fc1(torch.cat([cnn_out, non_spatial], dim=1)))
        h = self.rnn(x, hidden_state)
        return self.fc2(h), h


class QMixer(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_embed_dim=32, hypernet_embed=128):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * mixing_embed_dim))
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed_dim))
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim), nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(bs, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(bs)


# ============================================================================
# 5. MÔI TRƯỜNG
# ============================================================================
class UAVSwarmCoverageEnv:
    NOT_SCANNED = Map.CellState.NOT_SCANNED
    SCANNED = Map.CellState.SCANNED
    UNREACHABLE = Map.CellState.UNREACHABLE
    NO_INTEREST = Map.CellState.NO_INTEREST

    ACTIONS = [
        (0, -1), (0, 1), (-1, 0), (1, 0),     # lên, xuống, trái, phải
        (-1, -1), (-1, 1), (1, -1), (1, 1),   # chéo
    ]
    N_ACTIONS = 8

    def __init__(self, state, priority, uav_start, num_uavs=7, obs_radius=3,
                 max_steps=1500, step_penalty=0.01, invalid_penalty=0.1,
                 priority_weight=0.1, spawn_jitter=False,
                 allow_corner_cutting=True, seed=None,
                 use_energy=False, use_clusters=False, use_measurer=False,
                 r_collision=0.0, r_cluster_direction=0.0,
                 revisit_penalty=0.0, no_interest_penalty=0.0, shaping_coef=0.0):
        self.init_state = np.array(state, dtype=np.float32)
        self.priority = np.array(priority, dtype=np.float32)
        assert self.init_state.shape == self.priority.shape, \
            "state and priority grids must have the same shape"
        self.width, self.height = self.init_state.shape
        self.uav_start = tuple(uav_start)
        self.num_uavs = num_uavs
        self.obs_radius = obs_radius
        self.obs_window = obs_radius * 2 + 1
        spatial_dim = self.obs_window * self.obs_window
        self.obs_dim = spatial_dim * 3 + 2  # 3 kênh không gian + vector radar (dx, dy)
        self.state_dim = (self.width * self.height) * 2 + self.num_uavs * 2
        self.priority_scale = max(float(np.max(np.abs(self.priority), initial=0.0)), 1.0)
        self.normalized_priority = self.priority / self.priority_scale
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.invalid_penalty = invalid_penalty
        self.priority_weight = priority_weight
        self.r_collision = r_collision
        self.r_cluster_direction = r_cluster_direction
        self.revisit_penalty = revisit_penalty
        self.no_interest_penalty = no_interest_penalty
        self.shaping_coef = shaping_coef
        self.spawn_jitter = spawn_jitter
        self.allow_corner_cutting = allow_corner_cutting
        self.rng = np.random.RandomState(seed)

        # --- mô hình pin / bay về trạm sạc ---
        self.use_energy = use_energy
        if self.use_energy:
            self.obs_dim += 1
            self.state_dim += self.num_uavs
            self.max_energy = float(uav_distance)
            self.time_charge = int(time_charge)
            self.dis_threshold = float(dis_threshold)
            self.base = tuple(base)
            self.energy = np.full(self.num_uavs, self.max_energy, dtype=np.float32)
            self.charge_timer = np.zeros(self.num_uavs, dtype=np.int32)
            self.is_blocked = np.zeros(self.num_uavs, dtype=bool)
            self.return_path = [None] * self.num_uavs
            self._return_route_cache = {}

        # --- phủ theo cluster ---
        self.use_clusters = use_clusters
        self.clusters = []
        self.current_cluster_index = 0
        if self.use_clusters:
            self.clusters = calculate_centroid_priority(Map(self.init_state, self.priority))

        # --- ghi thời gian/chi phí ---
        self.use_measurer = use_measurer
        self.measurer = None

        self.reset()

    # ------------------------------------------------------------------ #
    def _can_move(self, pos, dx, dy, grid=None):
        grid = self.state if grid is None else grid
        x, y = pos
        nx, ny = x + dx, y + dy
        if not self._in_bounds(nx, ny) or grid[nx, ny] == self.UNREACHABLE:
            return False
        if dx != 0 and dy != 0 and not self.allow_corner_cutting:
            if (grid[x + dx, y] == self.UNREACHABLE or
                    grid[x, y + dy] == self.UNREACHABLE):
                return False
        return True

    def _spawn_positions(self):
        positions = [list(self.uav_start)]
        frontier = [self.uav_start]
        visited = {self.uav_start}
        while len(positions) < self.num_uavs and frontier:
            x, y = frontier.pop(0)
            for dx, dy in self.ACTIONS:
                nx, ny = x + dx, y + dy
                if ((nx, ny) not in visited and
                        self._can_move((x, y), dx, dy, self.init_state)):
                    visited.add((nx, ny))
                    frontier.append((nx, ny))
                    if len(positions) < self.num_uavs:
                        positions.append([nx, ny])
        while len(positions) < self.num_uavs:
            positions.append(list(self.uav_start))
        return positions

    def reset(self):
        self.state = self.init_state.copy()
        self.positions = self._spawn_positions() if self.spawn_jitter else \
            [list(self.uav_start) for _ in range(self.num_uavs)]
        self.steps = 0
        self.total_targets = int(np.sum(self.init_state == self.NOT_SCANNED))
        self.scanned = 0
        if self.use_energy:
            self.energy[:] = self.max_energy
            self.charge_timer[:] = 0
            self.is_blocked[:] = False
            self.return_path = [None] * self.num_uavs
        if self.use_clusters:
            self.current_cluster_index = 0
        if self.use_measurer:
            self.measurer = Measurer(type="qmix_v2", num_of_uavs=self.num_uavs, map=maptype)
        self._scan_current_cells()
        if self.use_measurer:
            self.measurer.get_data(self.scanned)
        return self.get_obs_all(), self.get_state()

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _return_route_to_base(self, position):
        """Đường tránh vật cản về trạm sạc; số bước khớp với mức tiêu hao pin."""
        position = tuple(position)
        cached = self._return_route_cache.get(position)
        if cached is None:
            path = find_path(
                position, self.base, Map(self.init_state, self.priority),
                allow_corner_cutting=self.allow_corner_cutting)
            expanded = expand_path_to_grid_steps(path) if path else None
            if not expanded:
                raise RuntimeError(
                    f"No obstacle-free return path from {position} to base {self.base}")
            cached = tuple(expanded)
            self._return_route_cache[position] = cached
        return cached

    def _current_valid_cells(self):
        if self.use_clusters and self.current_cluster_index < len(self.clusters):
            return self.clusters[self.current_cluster_index].available_cells
        return None

    def _nearest_unscanned_dist(self, pos):
        """GPS dẫn đường: khoảng cách Euclid tới ô chưa quét gần nhất trong cụm hiện tại."""
        x0, y0 = pos
        valid_cells = self._current_valid_cells()

        if self.state[x0, y0] == self.NOT_SCANNED:
            if valid_cells is None or (x0, y0) in valid_cells:
                return 0
        if valid_cells is None:
            return None

        min_dist = float('inf')
        found = False
        for cx, cy in valid_cells:
            if self.state[cx, cy] == self.NOT_SCANNED:
                dist = ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    found = True
        return min_dist if found else None

    def _scan_current_cells(self):
        reward = 0.0
        valid_cells = self._current_valid_cells()
        for x, y in self.positions:
            if self.state[x, y] == self.NOT_SCANNED:
                # Ô trắng nằm ngoài cụm hiện tại -> không tính quét, không thưởng
                if valid_cells is not None and (x, y) not in valid_cells:
                    continue
                self.state[x, y] = self.SCANNED
                reward += 1.0 + self.priority_weight * self.priority[x, y]
                self.scanned += 1
        return reward

    # ------------------------------------------------------------------ #
    def get_local_obs(self, pos, idx=None):
        x, y = pos
        r = self.obs_radius
        state_window = np.full(
            (self.obs_window, self.obs_window), self.UNREACHABLE, dtype=np.float32)
        priority_window = np.zeros((self.obs_window, self.obs_window), dtype=np.float32)
        x0, x1 = max(0, x - r), min(self.width, x + r + 1)
        y0, y1 = max(0, y - r), min(self.height, y + r + 1)
        wx0, wy0 = x0 - (x - r), y0 - (y - r)
        window_slice = (
            slice(wx0, wx0 + (x1 - x0)),
            slice(wy0, wy0 + (y1 - y0)),
        )

        state_window[window_slice] = self.state[x0:x1, y0:y1]
        priority_window[window_slice] = self.normalized_priority[x0:x1, y0:y1]

        # Kênh cluster mask
        cluster_window = np.zeros((self.obs_window, self.obs_window), dtype=np.float32)
        cluster_mask_global = np.zeros_like(self.state, dtype=np.float32)
        valid_cells = self._current_valid_cells()
        if valid_cells is not None:
            for cx, cy in valid_cells:
                cluster_mask_global[cx, cy] = 1.0
        cluster_window[window_slice] = cluster_mask_global[x0:x1, y0:y1]

        # Radar chỉ thẳng vào ô trắng gần nhất trong cụm
        target_x, target_y = x, y
        min_dist = float('inf')
        if valid_cells is not None:
            for cx, cy in valid_cells:
                if self.state[cx, cy] == self.NOT_SCANNED:
                    dist = (cx - x) ** 2 + (cy - y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        target_x, target_y = cx, cy
        pos_feat = np.array(
            [(target_x - x) / self.width, (target_y - y) / self.height], dtype=np.float32)

        feats = [
            state_window.flatten() / 3.0,
            priority_window.flatten(),
            cluster_window.flatten(),
            pos_feat,
        ]
        if self.use_energy:
            battery = self.energy[idx] / self.max_energy if idx is not None else 1.0
            feats.append(np.array([battery], dtype=np.float32))
        return np.concatenate(feats).astype(np.float32)

    def get_obs_all(self):
        return [self.get_local_obs(p, idx=i) for i, p in enumerate(self.positions)]

    def get_avail_actions(self, pos):
        x, y = pos
        avail = np.ones(self.N_ACTIONS, dtype=np.float32)
        for a, (dx, dy) in enumerate(self.ACTIONS):
            if not self._can_move((x, y), dx, dy):
                avail[a] = 0.0
        return avail

    def get_avail_actions_all(self):
        avail_all = []
        for i, pos in enumerate(self.positions):
            if self.use_energy and self.is_blocked[i]:
                # Môi trường điều khiển UAV khi đang về trạm/sạc: chỉ để 1 action giả
                avail = np.zeros(self.N_ACTIONS, dtype=np.float32)
                avail[0] = 1.0
                avail_all.append(avail)
            else:
                avail_all.append(self.get_avail_actions(pos))
        return avail_all

    def get_state(self):
        pos_feat = np.array(self.positions, dtype=np.float32)
        pos_feat[:, 0] /= self.width
        pos_feat[:, 1] /= self.height

        cluster_mask_global = np.zeros_like(self.state, dtype=np.float32)
        valid_cells = self._current_valid_cells()
        if valid_cells is not None and len(valid_cells) > 0:
            vx, vy = zip(*valid_cells)
            cluster_mask_global[vx, vy] = 1.0

        feats = [
            self.state.flatten() / 3.0,
            cluster_mask_global.flatten(),
            pos_feat.flatten(),
        ]
        if self.use_energy:
            feats.append(self.energy / self.max_energy)
        return np.concatenate(feats).astype(np.float32)

    # ------------------------------------------------------------------ #
    def step(self, actions):
        reward = -self.step_penalty

        prev_positions = [tuple(p) for p in self.positions]
        # UAV đang bị môi trường điều khiển (về trạm/sạc) -> không nhận shaping
        forced_transition = self.is_blocked.copy() if self.use_energy else \
            np.zeros(self.num_uavs, dtype=bool)

        old_dists = [
            None if forced_transition[i] else self._nearest_unscanned_dist(p)
            for i, p in enumerate(prev_positions)
        ] if self.shaping_coef != 0.0 else [None] * self.num_uavs

        cluster_target = None
        if self.use_clusters and self.current_cluster_index < len(self.clusters):
            ccx, ccy = self.clusters[self.current_cluster_index].center
            cluster_target = (ccx // cell_size, ccy // cell_size)

        if self.use_measurer:
            for (px, py) in prev_positions:
                self.measurer.add_cost(self.priority[px, py])
            self.measurer.tick_time()

        for i, a in enumerate(actions):
            if self.use_energy and self.is_blocked[i]:
                if self.return_path[i]:
                    # Đang bay về: tiến một waypoint, vẫn tốn pin
                    self.positions[i] = list(self.return_path[i].pop(0))
                    self.energy[i] -= 1.0
                else:
                    # Đã đáp xuống trạm: đếm ngược sạc, xong thì đầy pin
                    self.charge_timer[i] -= 1
                    if self.charge_timer[i] <= 0:
                        self.is_blocked[i] = False
                        self.energy[i] = self.max_energy
                        self.charge_timer[i] = self.time_charge
                        self.return_path[i] = None
                continue

            dx, dy = self.ACTIONS[a]
            x, y = self.positions[i]
            nx, ny = x + dx, y + dy
            moved = self._can_move((x, y), dx, dy)
            if moved:
                self.positions[i] = [nx, ny]
            else:
                reward -= self.invalid_penalty

            if self.use_energy and moved:
                self.energy[i] -= 1.0
                cur = tuple(self.positions[i])
                route = self._return_route_to_base(cur)
                return_steps = len(route) - 1
                if return_steps + 1 >= self.energy[i]:
                    # Pin chỉ đủ để về trạm -> chặn điều khiển, bay về sạc
                    self.is_blocked[i] = True
                    self.charge_timer[i] = self.time_charge
                    self.return_path[i] = list(route[1:])

        current_positions = [tuple(p) for p in self.positions]
        controllable_positions = [
            pos for i, pos in enumerate(current_positions) if not forced_transition[i]
        ]
        unique = len(set(controllable_positions))
        if unique < len(controllable_positions):
            reward += self.r_collision * (len(controllable_positions) - unique)

        if cluster_target is not None:
            tx, ty = cluster_target
            for i, ((cx, cy), (nx, ny)) in enumerate(zip(prev_positions, self.positions)):
                if forced_transition[i]:
                    continue
                dist_to_center = ((tx - nx) ** 2 + (ty - ny) ** 2) ** 0.5
                # Chỉ kéo về tâm cụm khi UAV còn ở ngoài bán kính cụm
                if dist_to_center > cell_radius:
                    old_dist = (((tx - cx) / self.width) ** 2 + ((ty - cy) / self.height) ** 2) ** 0.5
                    new_dist = (((tx - nx) / self.width) ** 2 + ((ty - ny) / self.height) ** 2) ** 0.5
                    reward += self.r_cluster_direction * (old_dist - new_dist)

        # Phạt theo ô vừa đáp xuống -- phải đọc TRƯỚC _scan_current_cells()
        valid_cells = self._current_valid_cells()
        for i, (px, py) in enumerate(current_positions):
            if forced_transition[i]:
                continue
            landed_state = self.state[px, py]
            if valid_cells is not None and (px, py) not in valid_cells:
                reward -= self.no_interest_penalty          # lấn sang cụm khác
            elif landed_state == self.SCANNED:
                reward -= self.revisit_penalty              # đi lại ô đã quét
            elif landed_state == self.NO_INTEREST:
                reward -= self.no_interest_penalty          # dẫm vùng xám

        if self.shaping_coef != 0.0:
            new_dists = [
                None if forced_transition[i] else self._nearest_unscanned_dist(p)
                for i, p in enumerate(current_positions)
            ]
            for old_d, new_d in zip(old_dists, new_dists):
                if old_d is not None and new_d is not None:
                    reward += self.shaping_coef * (old_d - new_d)

        scanned_before = self.scanned
        reward += self._scan_current_cells()
        if self.use_measurer:
            self.measurer.get_data(self.scanned - scanned_before)
        self.steps += 1

        if self.use_clusters and self.current_cluster_index < len(self.clusters):
            cx, cy = self.clusters[self.current_cluster_index].center
            if Map.is_cluster_scanned(self.state, Point(cx, cy), cell_radius):
                self.current_cluster_index += 1
        clusters_done = self.use_clusters and self.current_cluster_index >= len(self.clusters)

        done = self.scanned >= self.total_targets or self.steps >= self.max_steps or clusters_done
        coverage = self.scanned / max(1, self.total_targets)
        info = {"coverage": coverage}
        if self.use_clusters:
            info["cluster_index"] = min(self.current_cluster_index, len(self.clusters))
            info["num_clusters"] = len(self.clusters)

        if self.use_measurer and done:
            self.measurer.print()

        return self.get_obs_all(), self.get_state(), reward, done, info


# ============================================================================
# 6. RENDER (pygame, dùng lại Drawer.py)
# ============================================================================
# Drawer.draw_cluster_cells() đọc Map.cluster_cells (thuộc tính lớp); đặt mặc định
# rỗng để không lỗi khi use_clusters=False.
Map.cluster_cells = []


def _get_or_create_render_swarm(drawer, env):
    swarm = getattr(drawer, "_qmix_render_swarm", None)
    if swarm is None or len(swarm.uavs) != len(env.positions):
        uavs = []
        for i, (gx, gy) in enumerate(env.positions):
            pos = Point(gx * cell_size + cell_size // 2, gy * cell_size + cell_size // 2)
            uavs.append(UAV(
                distance=uav_distance,
                is_blocked=False,
                time_charge=time_charge,
                min_speed=min_speed[i % len(min_speed)],
                max_speed=max_speed[i % len(max_speed)],
                buffer_data=0,
                recent_position=pos,
                image_path="./images/uav.png",
                max_distance=env.max_energy if env.use_energy else None,
            ))
        swarm = Swarm(uavs, Point(0, 0))
        _update_swarm_center(swarm)
        drawer._qmix_render_swarm = swarm
    return swarm


def _sync_uav_energy(swarm, env):
    """Chép trạng thái pin thật từ env sang UAV vẽ để Drawer vẽ thanh năng lượng."""
    if not env.use_energy:
        return
    for uav, energy, blocked, pos, charge_timer in zip(
        swarm.uavs, env.energy, env.is_blocked, env.positions, env.charge_timer
    ):
        uav.distance = float(energy)
        uav.max_distance = env.max_energy
        uav.is_blocked = bool(blocked)
        uav.at_base = tuple(pos) == env.base
        uav.charging_progress = None
        if uav.is_blocked and uav.at_base and env.time_charge > 0:
            uav.charging_progress = max(
                0.0, min(1.0, 1.0 - float(charge_timer) / env.time_charge))


def _update_swarm_center(swarm):
    cx = sum(u.recent_position.x for u in swarm.uavs) / len(swarm.uavs)
    cy = sum(u.recent_position.y for u in swarm.uavs) / len(swarm.uavs)
    swarm.set_center(Point(int(cx), int(cy)))


def _snap_swarm_to_positions(swarm, positions):
    """Dịch chuyển tức thời (đầu episode, không có gì để nội suy)."""
    for uav, (gx, gy) in zip(swarm.uavs, positions):
        uav.recent_position.x = gx * cell_size + cell_size // 2
        uav.recent_position.y = gy * cell_size + cell_size // 2
        uav.recent_path = None
        uav.index_path = 0
        uav.status = UAV.UAVState.FREE
    _update_swarm_center(swarm)


def _set_swarm_targets(swarm, positions):
    for uav, (gx, gy) in zip(swarm.uavs, positions):
        uav.recent_path = [(gx, gy)]
        uav.index_path = 0
        uav.status = UAV.UAVState.BUSY


def render_env(drawer, env, fps=60):
    map_obj = Map(state=env.state.astype(int), priority=env.priority.astype(int))
    swarm = _get_or_create_render_swarm(drawer, env)
    _sync_uav_energy(swarm, env)

    circle_centers = []
    if env.use_clusters:
        for cluster in env.clusters:
            circle_centers.append((int(cluster.center[0]), int(cluster.center[1])))

    def _pump_and_draw():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                drawer.kill_window()
        _update_swarm_center(swarm)
        drawer.draw_all(map_obj, swarm, cir_centers=circle_centers)
        drawer.clock.tick(fps)

    if env.steps == 0:
        # Hiện các chấm hồng của cluster ngay từ bước đầu
        if env.use_clusters and env.current_cluster_index < len(env.clusters):
            cx, cy = env.clusters[env.current_cluster_index].center
            Map.is_cluster_scanned(env.state, Point(cx, cy), cell_radius)
        _snap_swarm_to_positions(swarm, env.positions)
        _pump_and_draw()
        return

    # Giữa episode: bay mượt từng UAV tới ô mới
    _set_swarm_targets(swarm, env.positions)
    max_frames = int(fps * 3) + 1
    frames = 0
    while any(uav.recent_path is not None for uav in swarm.uavs) and frames < max_frames:
        swarm.move_a_frame()
        _pump_and_draw()
        frames += 1


# ============================================================================
# 7. ROLLOUT + HUẤN LUYỆN
# ============================================================================
def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def select_actions(agent_net, obs_n, hidden_n, avail_n, n_agents, epsilon, device):
    """Epsilon-greedy cho TẤT CẢ agent trong một lần forward."""
    obs = torch.tensor(np.array(obs_n), dtype=torch.float32, device=device)
    agent_ids = F.one_hot(torch.arange(n_agents), n_agents).float().to(device)
    x = torch.cat([obs, agent_ids], dim=1)
    hidden = torch.cat(hidden_n, dim=0)

    with torch.no_grad():
        q, h = agent_net(x, hidden)
        avail = torch.tensor(np.array(avail_n), dtype=torch.float32, device=device)
        q = q.masked_fill(avail == 0, -1e9)
        greedy = torch.argmax(q, dim=1).tolist()

    actions = []
    for i in range(n_agents):
        if random.random() < epsilon:
            valid = np.nonzero(avail_n[i])[0]
            actions.append(int(random.choice(valid)) if len(valid) > 0 else greedy[i])
        else:
            actions.append(greedy[i])
    new_hidden = [h[i:i + 1] for i in range(n_agents)]
    return actions, new_hidden


def run_episode(env, agent_net, epsilon, device, n_agents, episode_idx=None,
                drawer=None, render_fps=60):
    obs_n, gstate = env.reset()
    hidden = [agent_net.init_hidden(1, device) for _ in range(n_agents)]
    episode = {"obs": [], "state": [], "avail": [], "actions": [], "reward": [], "done": []}
    done, ep_reward, info = False, 0.0, {"coverage": 0.0}

    if drawer:
        render_env(drawer, env, fps=render_fps)

    while not done:
        avail_n = env.get_avail_actions_all()
        actions, hidden = select_actions(agent_net, obs_n, hidden, avail_n, n_agents, epsilon, device)
        next_obs_n, next_state, reward, done, info = env.step(actions)

        episode["obs"].append(obs_n)
        episode["state"].append(gstate)
        episode["avail"].append(avail_n)
        episode["actions"].append(actions)
        episode["reward"].append(reward)
        episode["done"].append(done)

        obs_n, gstate = next_obs_n, next_state
        ep_reward += reward

        if drawer:
            render_env(drawer, env, fps=render_fps)

    episode["final_obs"] = obs_n
    episode["final_state"] = gstate
    episode["final_avail"] = env.get_avail_actions_all()
    return episode, ep_reward, info["coverage"]


def build_batch_tensors(batch, max_len, n_agents, obs_dim, n_actions, state_dim, device):
    bs = len(batch)
    obs = np.zeros((max_len, bs, n_agents, obs_dim), dtype=np.float32)
    avail = np.zeros((max_len, bs, n_agents, n_actions), dtype=np.float32)
    actions = np.zeros((max_len, bs, n_agents), dtype=np.int64)
    reward = np.zeros((max_len, bs), dtype=np.float32)
    done = np.zeros((max_len, bs), dtype=np.float32)
    mask = np.zeros((max_len, bs), dtype=np.float32)
    st = np.zeros((max_len, bs, state_dim), dtype=np.float32)
    obs_tp1 = np.zeros((max_len, bs, n_agents, obs_dim), dtype=np.float32)
    avail_tp1 = np.zeros((max_len, bs, n_agents, n_actions), dtype=np.float32)
    state_tp1 = np.zeros((max_len, bs, state_dim), dtype=np.float32)

    for b, ep in enumerate(batch):
        T = len(ep["reward"])
        mask[:T, b] = 1.0
        obs[:T, b] = np.asarray(ep["obs"], dtype=np.float32)
        avail[:T, b] = np.asarray(ep["avail"], dtype=np.float32)
        actions[:T, b] = np.asarray(ep["actions"], dtype=np.int64)
        reward[:T, b] = np.asarray(ep["reward"], dtype=np.float32)
        done[:T, b] = np.asarray(ep["done"], dtype=np.float32)
        st[:T, b] = np.asarray(ep["state"], dtype=np.float32)

        if T > 1:
            obs_tp1[:T - 1, b] = np.asarray(ep["obs"][1:], dtype=np.float32)
            avail_tp1[:T - 1, b] = np.asarray(ep["avail"][1:], dtype=np.float32)
            state_tp1[:T - 1, b] = np.asarray(ep["state"][1:], dtype=np.float32)
        obs_tp1[T - 1, b] = ep["final_obs"]
        avail_tp1[T - 1, b] = ep["final_avail"]
        state_tp1[T - 1, b] = ep["final_state"]

    def to_t(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype, device=device)

    return {
        "obs": to_t(obs), "avail": to_t(avail), "actions": to_t(actions, torch.int64),
        "reward": to_t(reward), "done": to_t(done), "mask": to_t(mask), "state": to_t(st),
        "obs_tp1": to_t(obs_tp1), "avail_tp1": to_t(avail_tp1), "state_tp1": to_t(state_tp1),
    }


def train_on_batch(batch, max_len, agent_net, target_agent_net, mixer, target_mixer,
                   optimizer, args, n_agents, n_actions, device):
    bs = len(batch)
    obs_dim = agent_net.obs_dim
    state_dim = mixer.state_dim

    data = build_batch_tensors(batch, max_len, n_agents, obs_dim, n_actions, state_dim, device)

    hidden = agent_net.init_hidden(bs * n_agents, device)
    target_hidden = target_agent_net.init_hidden(bs * n_agents, device)
    agent_ids = F.one_hot(torch.arange(n_agents), n_agents).float().to(device).repeat(bs, 1)

    seq_len = args.seq_len
    total_loss = 0.0
    chunks_trained = 0

    # Truncated BPTT: cắt episode thành các khúc seq_len bước
    for start_t in range(0, max_len, seq_len):
        q_tot_list, target_list, mask_list = [], [], []

        hidden = hidden.detach()
        target_hidden = target_hidden.detach()

        chunk_len = min(seq_len, max_len - start_t)

        for step in range(chunk_len):
            t = start_t + step
            mask_t = data["mask"][t]
            if mask_t.sum() == 0:
                break

            obs_t = data["obs"][t].reshape(bs * n_agents, obs_dim)
            x_t = torch.cat([obs_t, agent_ids], dim=1)
            q_t, hidden = agent_net(x_t, hidden)
            q_t = q_t.view(bs, n_agents, n_actions)
            actions_t = data["actions"][t].unsqueeze(-1)
            chosen_q = torch.gather(q_t, 2, actions_t).squeeze(-1)

            obs_tp1 = data["obs_tp1"][t].reshape(bs * n_agents, obs_dim)
            x_tp1 = torch.cat([obs_tp1, agent_ids], dim=1)
            with torch.no_grad():
                q_tp1_online, _ = agent_net(x_tp1, hidden.detach())
                _, target_hidden = target_agent_net(x_t, target_hidden)
                q_tp1_target, _ = target_agent_net(x_tp1, target_hidden)
                avail_tp1 = data["avail_tp1"][t]
                q_tp1_online = q_tp1_online.view(bs, n_agents, n_actions).masked_fill(avail_tp1 == 0, -1e9)
                best_actions = torch.argmax(q_tp1_online, dim=2, keepdim=True)
                q_tp1_target = q_tp1_target.view(bs, n_agents, n_actions)
                max_q_tp1 = torch.gather(q_tp1_target, 2, best_actions).squeeze(-1)

            q_tot = mixer(chosen_q, data["state"][t])
            with torch.no_grad():
                target_q_tot = target_mixer(max_q_tp1, data["state_tp1"][t])
                y = data["reward"][t] + args.gamma * (1 - data["done"][t]) * target_q_tot

            q_tot_list.append(q_tot)
            target_list.append(y)
            mask_list.append(mask_t)

        if len(q_tot_list) == 0:
            break

        q_tot_all = torch.stack(q_tot_list)
        target_all = torch.stack(target_list)
        mask_all = torch.stack(mask_list)

        chunk_mask_sum = mask_all.sum()
        if chunk_mask_sum > 0:
            loss = (F.huber_loss(q_tot_all, target_all, reduction='none') * mask_all).sum() / chunk_mask_sum

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent_net.parameters()) + list(mixer.parameters()), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            chunks_trained += 1

    # Soft update mạng mục tiêu
    for tp, p in zip(target_agent_net.parameters(), agent_net.parameters()):
        tp.data.copy_(tp.data * (1.0 - args.tau) + p.data * args.tau)
    for tp, p in zip(target_mixer.parameters(), mixer.parameters()):
        tp.data.copy_(tp.data * (1.0 - args.tau) + p.data * args.tau)

    return total_loss / max(1, chunks_trained)


# ============================================================================
# 8. CHECKPOINT (ghi nguyên tử, resume cả optimizer, RNG, epsilon)
# ============================================================================
CHECKPOINT_VERSION = 2
_CHECKPOINT_RUNTIME_ARGS = {
    "exp_name", "track", "wandb_project_name", "wandb_entity",
    "cuda", "render", "render_every", "render_fps",
    "save_model", "model_path", "resume", "save_every",
    "save_best", "best_max_steps", "best_score_floor",
}


def _model_spec(agent_net, mixer):
    return {
        "obs_dim": agent_net.obs_dim,
        "n_actions": agent_net.n_actions,
        "n_agents": agent_net.n_agents,
        "hidden_dim": agent_net.hidden_dim,
        "state_dim": mixer.state_dim,
        "mixing_embed_dim": mixer.embed_dim,
    }


def _warn_checkpoint_arg_mismatches(saved_args, args):
    if not saved_args or args is None:
        return
    current_args = asdict(args)
    mismatches = [
        (name, saved_value, current_args[name])
        for name, saved_value in saved_args.items()
        if (name in current_args and name not in _CHECKPOINT_RUNTIME_ARGS and
            saved_value != current_args[name])
    ]
    if mismatches:
        details = ", ".join(f"{n}: {s!r} -> {c!r}" for n, s, c in mismatches[:8])
        if len(mismatches) > 8:
            details += f", ... ({len(mismatches) - 8} more)"
        print(f"[QMIX] Warning: resumed configuration differs: {details}")


def save_checkpoint(path, agent_net, mixer, epsilon, episode,
                    target_agent_net=None, target_mixer=None, optimizer=None,
                    replay_buffer=None, env=None, args=None, next_epsilon=None):
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "agent": agent_net.state_dict(),
        "mixer": mixer.state_dict(),
        "eps": epsilon,
        "next_eps": next_epsilon,
        "episode": episode,
        "model_spec": _model_spec(agent_net, mixer),
    }
    if target_agent_net is not None:
        payload["target_agent"] = target_agent_net.state_dict()
    if target_mixer is not None:
        payload["target_mixer"] = target_mixer.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if replay_buffer is not None:
        payload["replay_buffer"] = {
            "capacity": replay_buffer.capacity,
            "episodes": replay_buffer.episodes,
        }
    if args is not None:
        payload["args"] = asdict(args)

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if env is not None:
        rng_state["env"] = env.rng.get_state()
    payload["rng_state"] = rng_state

    # Ghi ra file tạm rồi thay thế nguyên tử để không bao giờ có checkpoint dở dang
    path = os.fspath(path)
    temporary_path = f"{path}.tmp"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def load_checkpoint(path, agent_net, mixer, target_agent_net, target_mixer, device,
                    optimizer=None, replay_buffer=None, env=None, args=None):
    if not os.path.exists(path):
        print(f"[QMIX] No checkpoint at {path}, starting fresh.")
        return None, 0

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    checkpoint_version = ckpt.get("checkpoint_version", 1)
    if checkpoint_version > CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {checkpoint_version} is newer than supported "
            f"version {CHECKPOINT_VERSION}")

    saved_spec = ckpt.get("model_spec")
    current_spec = _model_spec(agent_net, mixer)
    if saved_spec is not None:
        differences = [
            f"{name}: checkpoint={saved_spec.get(name)!r}, current={value!r}"
            for name, value in current_spec.items()
            if saved_spec.get(name) != value]
        if differences:
            raise ValueError(
                "Checkpoint model configuration is incompatible (" +
                ", ".join(differences) + ")")

    agent_net.load_state_dict(ckpt["agent"])
    mixer.load_state_dict(ckpt["mixer"])
    if "target_agent" in ckpt:
        target_agent_net.load_state_dict(ckpt["target_agent"])
    else:
        target_agent_net.load_state_dict(agent_net.state_dict())
    if "target_mixer" in ckpt:
        target_mixer.load_state_dict(ckpt["target_mixer"])
    else:
        target_mixer.load_state_dict(mixer.state_dict())

    legacy_missing = []
    if optimizer is not None:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            legacy_missing.append("optimizer")
    if replay_buffer is not None:
        replay = ckpt.get("replay_buffer")
        if replay is not None:
            replay_buffer.episodes = list(replay["episodes"])[-replay_buffer.capacity:]
        else:
            legacy_missing.append("replay buffer")

    _warn_checkpoint_arg_mismatches(ckpt.get("args"), args)

    rng_state = ckpt.get("rng_state")
    if rng_state is not None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["torch_cuda"]])
            except RuntimeError as error:
                print(f"[QMIX] Warning: CUDA RNG state was not restored: {error}")
        if env is not None and "env" in rng_state:
            env.rng.set_state(rng_state["env"])
    else:
        legacy_missing.append("RNG state")

    start_episode = ckpt.get("episode", -1) + 1
    epsilon = ckpt.get("next_eps")
    if epsilon is None:
        if args is not None:
            epsilon = linear_schedule(
                args.start_e, args.end_e,
                args.exploration_fraction * args.total_episodes, start_episode)
        else:
            epsilon = ckpt.get("eps")
        legacy_missing.append("next-episode epsilon")

    if legacy_missing:
        print("[QMIX] Legacy checkpoint missing " + ", ".join(legacy_missing) +
              "; those parts were initialized for this run.")
    print(f"[QMIX] Loaded <- {path} (next episode={start_episode}, e={epsilon:.3f})")
    return epsilon, start_episode


# ============================================================================
# 9. LƯU BEST MODEL (thay cho watcher_save.py -- chạy ngay trong vòng huấn luyện)
# ============================================================================
_BEST_RE = re.compile(r"best_ep(\d+)_steps(\d+)_(-?\d+)\.pt$")


def _scan_best_files(directory="."):
    """Trả về list (score, đường_dẫn) của các best_*.pt đã có."""
    found = []
    for path in glob.glob(os.path.join(directory, "best_*.pt")):
        m = _BEST_RE.search(os.path.basename(path))
        if m:
            found.append((int(m.group(3)), path))
    return found


def find_best_checkpoint(fallback="qmix_model.pt", directory="."):
    """File best_*.pt có điểm cao nhất; không có thì dùng fallback."""
    found = _scan_best_files(directory)
    return max(found)[1] if found else fallback


class BestModelSaver:
    """Lưu best_ep{E}_steps{S}_{SCORE}.pt khi một episode:
        - coverage đạt 100%,
        - số bước < max_steps,
        - điểm (return) cao hơn kỷ lục hiện tại.
    Kỷ lục khởi tạo = max(score_floor, điểm cao nhất trong các best_*.pt đã có),
    nên chạy lại / resume sẽ không ghi đè bằng model kém hơn.
    Lưu thẳng trạng thái hiện tại vào file best (không phụ thuộc save_every).
    """

    def __init__(self, enabled=True, max_steps=1350, score_floor=9200.0, directory="."):
        self.enabled = enabled
        self.max_steps = max_steps
        self.directory = directory
        self.highest = float(score_floor)
        for score, _ in _scan_best_files(directory):
            self.highest = max(self.highest, float(score))

    def maybe_save(self, ep_idx, steps, ep_return, coverage, save_fn):
        if not self.enabled:
            return None
        if f"{coverage:.2%}" != "100.00%":      # khớp điều kiện cũ "coverage=100.00%"
            return None
        if steps >= self.max_steps or ep_return <= self.highest:
            return None

        name = os.path.join(self.directory, f"best_ep{ep_idx}_steps{steps}_{ep_return:.0f}.pt")
        save_fn(name)
        self.highest = ep_return
        print(f"\n[BEST] Episode {ep_idx}: steps={steps} | return={ep_return:.2f} -> {name}\n")
        return name


# ============================================================================
# 10. HÀM DỰNG ENV / MẠNG DÙNG CHUNG (test_demo.py cũng dùng)
# ============================================================================
def build_env(args):
    return UAVSwarmCoverageEnv(
        state, priority, uav_start, num_uavs=args.num_uavs, obs_radius=args.obs_radius,
        max_steps=args.max_steps_per_episode, step_penalty=args.step_penalty,
        invalid_penalty=args.invalid_penalty, priority_weight=args.priority_weight,
        spawn_jitter=args.spawn_jitter, allow_corner_cutting=args.allow_corner_cutting,
        seed=args.seed, use_energy=args.use_energy, use_clusters=args.use_clusters,
        use_measurer=args.use_measurer, r_collision=args.r_collision,
        r_cluster_direction=args.r_cluster_direction,
        revisit_penalty=args.revisit_penalty, no_interest_penalty=args.no_interest_penalty,
        shaping_coef=args.shaping_coef)


def build_networks(env, args, device):
    """Trả về (agent_net, target_agent_net, mixer, target_mixer); target = bản sao online."""
    agent_net = AgentQNetwork(env.obs_dim, env.N_ACTIONS, args.num_uavs, args.hidden_dim, env.obs_window).to(device)
    target_agent_net = AgentQNetwork(env.obs_dim, env.N_ACTIONS, args.num_uavs, args.hidden_dim, env.obs_window).to(device)
    target_agent_net.load_state_dict(agent_net.state_dict())

    mixer = QMixer(args.num_uavs, env.state_dim, args.mixing_embed_dim).to(device)
    target_mixer = QMixer(args.num_uavs, env.state_dim, args.mixing_embed_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())
    return agent_net, target_agent_net, mixer, target_mixer


# ============================================================================
# 11. MAIN
# ============================================================================
def main():
    args = tyro.cli(Args)
    if args.track:
        import wandb
        run_name = f"UAVCoverage__{args.exp_name}__{args.seed}__{int(time.time())}"
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   config=vars(args), name=run_name, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = build_env(args)
    print(f"Map: {env.width}x{env.height}  |  NOT_SCANNED cells to cover: {env.total_targets}  |  device: {device}")
    if args.use_clusters:
        print(f"Cluster-sequenced coverage enabled: {len(env.clusters)} clusters (cell_radius={cell_radius})")
    if args.use_measurer:
        print("Measurer logging enabled: type=qmix_v2, appending to output.txt every episode")
    if args.r_collision != 0.0 or args.r_cluster_direction != 0.0:
        print(f"Reward shaping: r_collision={args.r_collision}  r_cluster_direction={args.r_cluster_direction}")
    if args.revisit_penalty != 0.0 or args.no_interest_penalty != 0.0 or args.shaping_coef != 0.0:
        print(f"Reward shaping: revisit_penalty={args.revisit_penalty}  "
              f"no_interest_penalty={args.no_interest_penalty}  shaping_coef={args.shaping_coef}")

    agent_net, target_agent_net, mixer, target_mixer = build_networks(env, args, device)
    optimizer = torch.optim.Adam(
        list(agent_net.parameters()) + list(mixer.parameters()), lr=args.learning_rate)
    buffer = EpisodeBuffer(capacity=args.buffer_size)

    start_episode = 0
    resumed_eps = None
    if args.resume:
        resumed_eps, start_episode = load_checkpoint(
            args.model_path, agent_net, mixer, target_agent_net, target_mixer, device,
            optimizer=optimizer, replay_buffer=buffer, env=env, args=args)

    best_saver = BestModelSaver(
        enabled=args.save_best, max_steps=args.best_max_steps, score_floor=args.best_score_floor)
    if args.save_best:
        print(f"Best-model saver: 100% coverage & steps < {args.best_max_steps} "
              f"& return > {best_saver.highest:.0f}")

    def save_to(path, episode_idx, eps, next_eps):
        save_checkpoint(
            path, agent_net, mixer, eps, episode_idx,
            target_agent_net=target_agent_net, target_mixer=target_mixer,
            optimizer=optimizer, replay_buffer=None, env=env, args=args,
            next_epsilon=next_eps)

    start_time = time.time()
    drawer = Drawer("qmix") if args.render else None
    if drawer is not None:
        print(f"Rendering every {args.render_every} episode(s), up to {args.render_fps} FPS "
              f"-- close the window to stop the run.")

    epsilon = resumed_eps if resumed_eps is not None else linear_schedule(
        args.start_e, args.end_e,
        args.exploration_fraction * args.total_episodes, start_episode)
    ep_idx = start_episode - 1
    last_completed_episode = None
    last_episode_epsilon = None
    last_saved_episode = None

    for ep_idx in range(start_episode, args.total_episodes):
        do_render = drawer is not None and ep_idx % args.render_every == 0
        episode, ep_reward, coverage = run_episode(
            env, agent_net, epsilon, device, args.num_uavs, episode_idx=ep_idx,
            drawer=drawer if do_render else None, render_fps=args.render_fps)
        buffer.add(episode)
        ep_steps = len(episode["reward"])

        if args.track:
            wandb.log({
                "charts/episodic_return": ep_reward,
                "charts/coverage": coverage,
                "charts/epsilon": epsilon,
                "charts/episode_length": ep_steps,
            }, step=ep_idx)

        log_str = (f"episode={ep_idx:5d}  steps={ep_steps:4d}  return={ep_reward:9.2f}  "
                   f"coverage={coverage:6.2%}  epsilon={epsilon:.3f}")
        print(log_str)
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

        if len(buffer) >= args.learning_starts and ep_idx % args.train_every == 0:
            batch, max_len = buffer.sample(args.batch_size)
            loss = train_on_batch(batch, max_len, agent_net, target_agent_net, mixer,
                                  target_mixer, optimizer, args, args.num_uavs, env.N_ACTIONS, device)
            if args.track:
                wandb.log({"losses/qmix_loss": loss}, step=ep_idx)

        next_epsilon = linear_schedule(
            args.start_e, args.end_e,
            args.exploration_fraction * args.total_episodes, ep_idx + 1)

        if args.save_model and (ep_idx + 1) % args.save_every == 0:
            save_to(args.model_path, ep_idx, epsilon, next_epsilon)
            last_saved_episode = ep_idx

        # Lưu best model (sau khi đã train xong episode này)
        best_saver.maybe_save(
            ep_idx, ep_steps, ep_reward, coverage,
            save_fn=lambda p, i=ep_idx, e=epsilon, n=next_epsilon: save_to(p, i, e, n))

        last_completed_episode = ep_idx
        last_episode_epsilon = epsilon
        epsilon = next_epsilon

    if (args.save_model and last_completed_episode is not None and
            last_saved_episode != last_completed_episode):
        save_to(args.model_path, last_completed_episode, last_episode_epsilon, epsilon)

    print(f"Done in {time.time() - start_time:.1f}s")
    if drawer is not None:
        pygame.quit()
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()

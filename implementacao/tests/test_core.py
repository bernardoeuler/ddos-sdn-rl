import socket
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ddosrl.common import (N_ACTIONS, N_LOCAL, OBS_DIM, Action, RewardTerms,  # noqa: E402
                           load_config, reward)
from ddosrl.features import FeatureExtractor, GlobalSample, SourceSample, entropy_norm  # noqa: E402
from ddosrl.policies import FixedThreshold, RandomForestPolicy, I_PPS  # noqa: E402
from ddosrl.sdn.layout import layout  # noqa: E402
from ddosrl.sdn.sflow import build_datagram, parse_datagram  # noqa: E402
from ddosrl.sim.env import DDoSSimEnv  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def tcp_header(src, dst, sport, dport, flags):
    eth = b"\x00" * 12 + struct.pack("!H", 0x0800)
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 40, 0, 0, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 0, 0, 0x50, flags, 1024, 0, 0)
    return eth + ip + tcp


def test_sflow_roundtrip():
    hdrs = [tcp_header("10.0.0.3", "10.0.0.100", 1234, 80, 0x02),
            tcp_header("10.0.0.4", "10.0.0.100", 5555, 80, 0x12)]
    s = parse_datagram(build_datagram(hdrs, rate=64))
    assert len(s) == 2
    assert s[0].src == "10.0.0.3" and s[0].dport == 80 and s[0].is_syn
    assert not s[1].is_syn                       # SYN+ACK não conta como SYN
    assert s[0].sampling_rate == 64


def test_sflow_garbage_is_ignored():
    assert parse_datagram(b"\x00\x00\x00\x05garbage") == []


def test_features_ranges(cfg):
    fx = FeatureExtractor(cfg)
    for k in range(5):
        fx.push("a", SourceSample(pkts_out=8000 * k, bytes_out=8000 * k * 60, bytes_in=10,
                                  n_samples=10, n_syn=10, dst_ports={80}, conns=set(range(10))))
        fx.push("b", SourceSample(pkts_out=20, bytes_out=900, bytes_in=21000, n_samples=1, n_syn=0))
    xa, xb = fx.local("a"), fx.local("b")
    assert xa.shape == (N_LOCAL,)
    assert np.all(xa[:-1] >= 0) and np.all(xa[:-1] <= 1) and -1 <= xa[-1] <= 1
    assert xa[3] == 1.0 and xb[3] == 0.0         # proporção de SYN
    assert xb[8] > 0.9                           # cliente legítimo recebe respostas
    g = fx.global_(GlobalSample(victim_bytes=1.25e6, rtt_ms=20), n_sources=2)
    assert 0.99 < g[0] <= 1.0                    # 10 Mbit/s no gargalo de 10 Mbit/s
    obs = fx.observation("a", g, Action.LIMIT, 3.0)
    assert obs.shape == (OBS_DIM,)


def test_entropy():
    assert entropy_norm({"a": 10, "b": 10}, 2) == pytest.approx(1.0)
    assert entropy_norm({"a": 100, "b": 0}, 2) == pytest.approx(0.0)


def test_reward_penalizes_false_positive_more(cfg):
    t = RewardTerms(availability=1.0, attack_util=0.0)
    r_atk = reward(cfg, Action.BLOCK, Action.BLOCK, True, t)
    r_leg = reward(cfg, Action.BLOCK, Action.BLOCK, False, t)
    base = reward(cfg, Action.ALLOW, Action.ALLOW, False, t)
    assert r_atk - base > 0 and base - r_leg > r_atk - base


def test_graduated_actions_optimal_under_uncertainty(cfg):
    """Com P(ataque)=0,5 a melhor ação local esperada é intermediária."""
    t = RewardTerms(1.0, 0.0)
    exp = [0.5 * reward(cfg, a, a, True, t) + 0.5 * reward(cfg, a, a, False, t) for a in Action]
    assert int(np.argmax(exp)) in (Action.LIMIT, Action.REDIRECT)


def test_env_block_lock(cfg):
    env = DDoSSimEnv(cfg)
    obs, info = env.reset(seed=3, options={"scenario": "S1H"})
    assert obs.shape == (cfg["network"]["n_hosts"], OBS_DIM)
    acts = np.zeros(env.n, int)
    acts[0] = Action.BLOCK
    _, r, _, _, info = env.step(acts)
    assert r.shape == (env.n,)
    steps = int(cfg["agent"]["block_s"] / cfg["agent"]["dt_s"])
    for _ in range(steps - 1):
        assert not info["decide"][0]             # origem bloqueada não decide
        _, _, _, _, info = env.step(np.zeros(env.n, int))
    assert info["decide"][0] and env.src[0].action == Action.ALLOW   # hard_timeout expirou


def test_env_attack_reduces_availability(cfg):
    env = DDoSSimEnv(cfg)
    def avail(scen):
        env.reset(seed=11, options={"scenario": scen})
        a = []
        for _ in range(env.T):
            *_, info = env.step(np.zeros(env.n, int))
            if env.t_on <= env.t < env.t_off:
                a.append(info["availability"])
        return np.mean(a)
    assert avail("S2H") < avail("S0") - 0.2


def test_numpy_policy_matches_torch(cfg, tmp_path):
    torch = pytest.importorskip("torch")
    from ddosrl.agent.ddqn import DDQNAgent
    from ddosrl.agent.numpy_policy import NumpyDuelingQ
    ag = DDQNAgent(cfg, seed=0)
    ag.save(tmp_path / "m")
    x = np.random.default_rng(0).random((7, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        q_t = ag.q(torch.as_tensor(x)).numpy()
    q_n = NumpyDuelingQ(tmp_path / "m").q_values(x)
    assert np.allclose(q_t, q_n, atol=1e-5)


def test_smdp_collector_discounts():
    from train_agent import SMDPCollector
    from ddosrl.agent.ddqn import ReplayBuffer
    buf = ReplayBuffer(10)
    c = SMDPCollector(1, gamma=0.5)
    s0, s1 = np.zeros(OBS_DIM), np.ones(OBS_DIM)
    c.decide(0, s0, 3, buf)
    for _ in range(3):
        c.reward([1.0])
    c.decide(0, s1, 0, buf)
    assert len(buf) == 1
    assert buf.r[0] == pytest.approx(1 + 0.5 + 0.25)
    assert buf.g[0] == pytest.approx(0.125)


def test_baselines():
    obs = np.zeros((3, OBS_DIM), np.float32)
    obs[1, I_PPS] = 0.5
    assert list(FixedThreshold(0.1).decide(obs)) == [0, Action.BLOCK, 0]

    class FakeRF:
        def predict_proba(self, X):
            p = np.array([0.1, 0.6, 0.8, 0.95])
            return np.stack([1 - p, p], 1)
    acts = RandomForestPolicy(FakeRF(), graduated=True).decide(np.zeros((4, OBS_DIM)))
    assert list(acts) == [Action.ALLOW, Action.LIMIT, Action.REDIRECT, Action.BLOCK]
    assert N_ACTIONS == 4


def test_layout_ports_unique(cfg):
    L = layout(cfg)
    core_ports = [L[k]["port"] for k in ("victim", "analysis", "monitor")] + list(L["core_port_of_edge"].values())
    assert len(set(core_ports)) == len(core_ports)
    for dpid in L["edge_dpids"]:
        ports = [h["port"] for h in L["hosts"].values() if h["dpid"] == dpid]
        assert len(set(ports)) == len(ports) and L["uplink_port"] not in ports

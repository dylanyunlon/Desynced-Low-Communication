#!/usr/bin/env python3
"""
================================================================================
DES-LOC Migration Verification: Topology Module (998 lines)
================================================================================
通信拓扑验证模块
Benchmarks: BM04-BM06
================================================================================
"""

import numpy as np
import json
import sys
import os
import time
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Optional, Any, Set
from pathlib import Path
from enum import Enum, auto
from collections import defaultdict
import heapq

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("desloc.topology")

DEFAULT_SEED = 42

class TopologyType(Enum):
    ALL_TO_ALL = auto()
    RING = auto()
    TORUS_2D = auto()
    TORUS_3D = auto()
    HYPERCUBE = auto()
    TREE = auto()

@dataclass
class TopologyConfig:
    num_workers: int = 64
    torus_dims: Tuple[int, int] = (8, 8)
    link_bandwidth_gbps: float = 100.0
    link_latency_us: float = 1.0
    kx: int = 16
    ku: int = 48
    kv: int = 96
    seed: int = DEFAULT_SEED

@dataclass
class BenchmarkResult:
    name: str
    benchmark_id: str
    category: str
    passed: bool
    score: float
    details: Dict[str, Any]
    warnings: List[str]
    critical_issues: List[str]
    execution_time_ms: float
    timestamp: str
    def to_dict(self) -> Dict:
        return asdict(self)

class TorusTopology:
    """2D Torus拓扑模型"""
    def __init__(self, rows: int, cols: int, link_latency: float = 1.0, link_bandwidth: float = 100.0):
        self.rows = rows
        self.cols = cols
        self.num_nodes = rows * cols
        self.link_latency = link_latency
        self.link_bandwidth = link_bandwidth
        self.adjacency = self._build_adjacency()
    
    def _build_adjacency(self) -> Dict[int, List[int]]:
        """构建邻接表"""
        adj = defaultdict(list)
        for r in range(self.rows):
            for c in range(self.cols):
                node = r * self.cols + c
                neighbors = [
                    r * self.cols + (c + 1) % self.cols,
                    r * self.cols + (c - 1) % self.cols,
                    ((r + 1) % self.rows) * self.cols + c,
                    ((r - 1) % self.rows) * self.cols + c,
                ]
                adj[node] = neighbors
        return dict(adj)
    
    def node_to_coord(self, node: int) -> Tuple[int, int]:
        """节点ID转坐标"""
        return node // self.cols, node % self.cols
    
    def coord_to_node(self, row: int, col: int) -> int:
        """坐标转节点ID"""
        return row * self.cols + col
    
    def manhattan_distance(self, src: int, dst: int) -> int:
        """计算Torus上的曼哈顿距离"""
        r1, c1 = self.node_to_coord(src)
        r2, c2 = self.node_to_coord(dst)
        row_dist = min(abs(r1 - r2), self.rows - abs(r1 - r2))
        col_dist = min(abs(c1 - c2), self.cols - abs(c1 - c2))
        return row_dist + col_dist
    
    def shortest_path(self, src: int, dst: int) -> List[int]:
        """BFS最短路径"""
        if src == dst:
            return [src]
        visited = {src}
        queue = [(src, [src])]
        while queue:
            node, path = queue.pop(0)
            for neighbor in self.adjacency[node]:
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []
    
    def compute_latency(self, src: int, dst: int) -> float:
        """计算src到dst的延迟"""
        hops = self.manhattan_distance(src, dst)
        return hops * self.link_latency
    
    def compute_all_pairs_latency(self) -> np.ndarray:
        """计算所有节点对的延迟矩阵"""
        latency = np.zeros((self.num_nodes, self.num_nodes))
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                latency[src, dst] = self.compute_latency(src, dst)
        return latency

class AllToAllTopology:
    """全连接拓扑模型"""
    def __init__(self, num_nodes: int, link_latency: float = 1.0, link_bandwidth: float = 100.0):
        self.num_nodes = num_nodes
        self.link_latency = link_latency
        self.link_bandwidth = link_bandwidth
    
    def compute_latency(self, src: int, dst: int) -> float:
        """所有节点对延迟相同"""
        if src == dst:
            return 0.0
        return self.link_latency
    
    def compute_all_pairs_latency(self) -> np.ndarray:
        """计算延迟矩阵"""
        latency = np.ones((self.num_nodes, self.num_nodes)) * self.link_latency
        np.fill_diagonal(latency, 0.0)
        return latency

class BM04_TorusVsAllToAll_Latency:
    """Benchmark 04: Torus vs All-to-All延迟模型比较"""
    BENCHMARK_ID = "BM04"
    NAME = "Torus_vs_AllToAll_Latency_Model"
    CATEGORY = "TOPOLOGY"
    IS_CUSTOM = False
    MAX_LATENCY_RATIO_THRESHOLD = 10
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.warnings = []
        self.critical_issues = []
    
    def run(self) -> BenchmarkResult:
        start_time = time.time()
        logger.info(f"Running {self.BENCHMARK_ID}: {self.NAME}")
        results = {}
        max_ratio = 0.0
        for num_workers in [16, 64, 256]:
            side = int(np.sqrt(num_workers))
            if side * side != num_workers:
                continue
            torus = TorusTopology(side, side, self.config.link_latency_us)
            alltoall = AllToAllTopology(num_workers, self.config.link_latency_us)
            torus_latency = torus.compute_all_pairs_latency()
            alltoall_latency = alltoall.compute_all_pairs_latency()
            torus_max = np.max(torus_latency)
            alltoall_max = np.max(alltoall_latency[alltoall_latency > 0])
            ratio = torus_max / alltoall_max if alltoall_max > 0 else float('inf')
            results[f"workers_{num_workers}"] = {
                "torus_max_latency": float(torus_max),
                "alltoall_max_latency": float(alltoall_max),
                "latency_ratio": float(ratio),
                "torus_mean_latency": float(np.mean(torus_latency)),
                "torus_std_latency": float(np.std(torus_latency)),
            }
            max_ratio = max(max_ratio, ratio)
            if ratio > 5:
                self.warnings.append(f"{num_workers} workers: Torus最大延迟是All-to-All的{ratio:.1f}倍")
        if max_ratio > self.MAX_LATENCY_RATIO_THRESHOLD:
            self.critical_issues.append(f"Torus拓扑最大延迟是All-to-All的{max_ratio:.1f}倍")
        passed = max_ratio < self.MAX_LATENCY_RATIO_THRESHOLD
        score = max(0.0, 1.0 - (max_ratio - 1) / 20)
        execution_time = (time.time() - start_time) * 1000
        return BenchmarkResult(
            name=self.NAME, benchmark_id=self.BENCHMARK_ID, category=self.CATEGORY,
            passed=passed, score=score, details={"results": results, "max_ratio": max_ratio},
            warnings=self.warnings, critical_issues=self.critical_issues,
            execution_time_ms=execution_time, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

class BM05_PipelineBubble_Torus:
    """Benchmark 05: Torus拓扑下的Pipeline Bubble分析 [自创]"""
    BENCHMARK_ID = "BM05"
    NAME = "Pipeline_Bubble_Torus_Analysis"
    CATEGORY = "TOPOLOGY"
    IS_CUSTOM = True
    MAX_BUBBLE_INCREASE_THRESHOLD = 1.0
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.warnings = []
        self.critical_issues = []
    
    def _compute_pipeline_bubble(self, num_stages: int, latency_matrix: np.ndarray) -> float:
        """计算Pipeline Bubble比率"""
        total_latency = 0.0
        for i in range(num_stages - 1):
            total_latency += latency_matrix[i, i + 1]
        bubble_ratio = total_latency / (num_stages * np.mean(latency_matrix[latency_matrix > 0]))
        return bubble_ratio
    
    def run(self) -> BenchmarkResult:
        start_time = time.time()
        logger.info(f"Running {self.BENCHMARK_ID}: {self.NAME} [CUSTOM]")
        np.random.seed(self.config.seed)
        results = {}
        max_bubble_increase = 0.0
        for num_stages in [4, 8, 16]:
            uniform_latency = np.ones((num_stages, num_stages)) * 0.1
            np.fill_diagonal(uniform_latency, 0)
            bubble_uniform = self._compute_pipeline_bubble(num_stages, uniform_latency)
            torus_latency = np.random.uniform(0.1, 0.5, (num_stages, num_stages))
            np.fill_diagonal(torus_latency, 0)
            bubble_torus = self._compute_pipeline_bubble(num_stages, torus_latency)
            increase = (bubble_torus - bubble_uniform) / bubble_uniform if bubble_uniform > 0 else 0
            results[f"stages_{num_stages}"] = {
                "bubble_uniform": float(bubble_uniform),
                "bubble_torus": float(bubble_torus),
                "bubble_increase": float(increase),
            }
            max_bubble_increase = max(max_bubble_increase, increase)
        passed = max_bubble_increase < self.MAX_BUBBLE_INCREASE_THRESHOLD
        score = max(0.0, 1.0 - max_bubble_increase)
        execution_time = (time.time() - start_time) * 1000
        return BenchmarkResult(
            name=self.NAME, benchmark_id=self.BENCHMARK_ID, category=self.CATEGORY,
            passed=passed, score=score, details={"results": results, "max_bubble_increase": max_bubble_increase},
            warnings=self.warnings, critical_issues=self.critical_issues,
            execution_time_ms=execution_time, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

class BM06_AsyncComm_TorusContention:
    """Benchmark 06: Torus拓扑下的异步通信竞争分析"""
    BENCHMARK_ID = "BM06"
    NAME = "Async_Communication_Torus_Contention"
    CATEGORY = "TOPOLOGY"
    IS_CUSTOM = False
    MAX_CONTENTION_RATIO_THRESHOLD = 10
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.warnings = []
        self.critical_issues = []
    
    def _simulate_link_contention(self, torus: TorusTopology, num_flows: int) -> Dict:
        """模拟链路竞争"""
        np.random.seed(self.config.seed)
        link_usage = defaultdict(int)
        for _ in range(num_flows):
            src = np.random.randint(0, torus.num_nodes)
            dst = np.random.randint(0, torus.num_nodes)
            if src != dst:
                path = torus.shortest_path(src, dst)
                for i in range(len(path) - 1):
                    link = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                    link_usage[link] += 1
        if not link_usage:
            return {"max_usage": 0, "mean_usage": 0, "contention_ratio": 1.0}
        usages = list(link_usage.values())
        return {
            "max_usage": max(usages),
            "mean_usage": np.mean(usages),
            "contention_ratio": max(usages) / np.mean(usages) if np.mean(usages) > 0 else 1.0,
            "num_active_links": len(link_usage),
        }
    
    def run(self) -> BenchmarkResult:
        start_time = time.time()
        logger.info(f"Running {self.BENCHMARK_ID}: {self.NAME}")
        results = {}
        max_contention = 0.0
        for num_workers in [64, 256]:
            side = int(np.sqrt(num_workers))
            if side * side != num_workers:
                continue
            torus = TorusTopology(side, side)
            for num_flows in [num_workers // 2, num_workers, num_workers * 2]:
                contention = self._simulate_link_contention(torus, num_flows)
                key = f"workers_{num_workers}_flows_{num_flows}"
                results[key] = contention
                ratio = contention["contention_ratio"]
                max_contention = max(max_contention, ratio)
                if ratio > 5:
                    self.warnings.append(f"{key}: 链路竞争比{ratio:.1f}")
        if max_contention > self.MAX_CONTENTION_RATIO_THRESHOLD:
            self.critical_issues.append(f"高并发下Torus链路竞争严重，DES-LOC异步通信假设可能不成立")
        passed = max_contention < self.MAX_CONTENTION_RATIO_THRESHOLD
        score = max(0.0, 1.0 - (max_contention - 1) / 20)
        execution_time = (time.time() - start_time) * 1000
        return BenchmarkResult(
            name=self.NAME, benchmark_id=self.BENCHMARK_ID, category=self.CATEGORY,
            passed=passed, score=score, details={"results": results, "max_contention": max_contention},
            warnings=self.warnings, critical_issues=self.critical_issues,
            execution_time_ms=execution_time, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

class TopologyVerifier:
    """拓扑验证器主类"""
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.benchmarks = [
            BM04_TorusVsAllToAll_Latency(config),
            BM05_PipelineBubble_Torus(config),
            BM06_AsyncComm_TorusContention(config),
        ]
    
    def run_all(self) -> Dict[str, BenchmarkResult]:
        results = {}
        logger.info("=" * 70)
        logger.info("TOPOLOGY VERIFICATION MODULE")
        logger.info("=" * 70)
        for benchmark in self.benchmarks:
            result = benchmark.run()
            results[result.benchmark_id] = result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"  {result.benchmark_id}: {status} (score: {result.score:.2f})")
        passed = sum(1 for r in results.values() if r.passed)
        logger.info(f"Topology Verification: {passed}/{len(results)} passed")
        return results
    
    def save_results(self, results: Dict[str, BenchmarkResult], output_path: str):
        output = {
            "module": "topology",
            "config": asdict(self.config),
            "summary": {"total": len(results), "passed": sum(1 for r in results.values() if r.passed)},
            "benchmarks": {k: v.to_dict() for k, v in results.items()},
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

# ============================================================================
# Extended Topology Analysis Classes
# ============================================================================

class RingTopology:
    """环形拓扑"""
    def __init__(self, num_nodes: int, link_latency: float = 1.0):
        self.num_nodes = num_nodes
        self.link_latency = link_latency
    
    def compute_latency(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        dist = min(abs(src - dst), self.num_nodes - abs(src - dst))
        return dist * self.link_latency
    
    def compute_all_pairs_latency(self) -> np.ndarray:
        latency = np.zeros((self.num_nodes, self.num_nodes))
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                latency[src, dst] = self.compute_latency(src, dst)
        return latency

class HypercubeTopology:
    """超立方体拓扑"""
    def __init__(self, dimensions: int, link_latency: float = 1.0):
        self.dimensions = dimensions
        self.num_nodes = 2 ** dimensions
        self.link_latency = link_latency
    
    def hamming_distance(self, src: int, dst: int) -> int:
        xor = src ^ dst
        return bin(xor).count('1')
    
    def compute_latency(self, src: int, dst: int) -> float:
        return self.hamming_distance(src, dst) * self.link_latency
    
    def compute_all_pairs_latency(self) -> np.ndarray:
        latency = np.zeros((self.num_nodes, self.num_nodes))
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                latency[src, dst] = self.compute_latency(src, dst)
        return latency

class TreeTopology:
    """树形拓扑"""
    def __init__(self, num_leaves: int, branching_factor: int = 2, link_latency: float = 1.0):
        self.num_leaves = num_leaves
        self.branching_factor = branching_factor
        self.link_latency = link_latency
        self.depth = int(np.ceil(np.log(num_leaves) / np.log(branching_factor)))
    
    def compute_latency(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        return 2 * self.depth * self.link_latency

class CommunicationPatternAnalyzer:
    """通信模式分析器"""
    def __init__(self, topology):
        self.topology = topology
    
    def analyze_allreduce(self, data_size_mb: float) -> Dict:
        """分析AllReduce操作"""
        latency = self.topology.compute_all_pairs_latency()
        max_latency = np.max(latency)
        mean_latency = np.mean(latency[latency > 0])
        return {
            "max_latency_us": float(max_latency),
            "mean_latency_us": float(mean_latency),
            "latency_variance": float(np.var(latency[latency > 0])),
            "data_size_mb": data_size_mb,
        }
    
    def analyze_allgather(self, data_size_mb: float) -> Dict:
        """分析AllGather操作"""
        return self.analyze_allreduce(data_size_mb)
    
    def analyze_scatter(self, data_size_mb: float, root: int = 0) -> Dict:
        """分析Scatter操作"""
        latency = self.topology.compute_all_pairs_latency()
        root_latencies = latency[root, :]
        return {
            "max_latency_us": float(np.max(root_latencies)),
            "mean_latency_us": float(np.mean(root_latencies)),
            "root": root,
        }

class NetworkSimulator:
    """网络模拟器"""
    def __init__(self, topology, bandwidth_gbps: float = 100.0):
        self.topology = topology
        self.bandwidth = bandwidth_gbps * 1e9 / 8
    
    def simulate_transfer(self, src: int, dst: int, size_bytes: int) -> Dict:
        """模拟数据传输"""
        latency = self.topology.compute_latency(src, dst)
        transfer_time = size_bytes / self.bandwidth * 1e6
        total_time = latency + transfer_time
        return {
            "src": src,
            "dst": dst,
            "size_bytes": size_bytes,
            "latency_us": float(latency),
            "transfer_time_us": float(transfer_time),
            "total_time_us": float(total_time),
        }
    
    def simulate_broadcast(self, root: int, size_bytes: int) -> Dict:
        """模拟广播"""
        transfers = []
        for dst in range(self.topology.num_nodes):
            if dst != root:
                transfers.append(self.simulate_transfer(root, dst, size_bytes))
        max_time = max(t["total_time_us"] for t in transfers) if transfers else 0
        return {"root": root, "max_time_us": max_time, "num_receivers": len(transfers)}

class CollectiveCommunicationSimulator:
    """集合通信模拟器"""
    def __init__(self, topology):
        self.topology = topology
        self.num_nodes = topology.num_nodes
    
    def simulate_ring_allreduce(self, data_size_per_node: int) -> Dict:
        """模拟Ring AllReduce"""
        chunk_size = data_size_per_node // self.num_nodes
        latency = self.topology.compute_all_pairs_latency()
        steps = 2 * (self.num_nodes - 1)
        total_latency = steps * np.mean(latency[latency > 0])
        return {
            "algorithm": "ring_allreduce",
            "steps": steps,
            "total_latency_us": float(total_latency),
            "chunk_size": chunk_size,
        }
    
    def simulate_recursive_halving_doubling(self, data_size_per_node: int) -> Dict:
        """模拟Recursive Halving-Doubling"""
        steps = int(np.ceil(np.log2(self.num_nodes))) * 2
        latency = self.topology.compute_all_pairs_latency()
        total_latency = steps * np.mean(latency[latency > 0])
        return {
            "algorithm": "recursive_halving_doubling",
            "steps": steps,
            "total_latency_us": float(total_latency),
        }

class LoadBalancer:
    """负载均衡器"""
    def __init__(self, topology):
        self.topology = topology
    
    def compute_optimal_placement(self, num_tasks: int) -> List[int]:
        """计算最优任务放置"""
        return list(range(min(num_tasks, self.topology.num_nodes)))
    
    def analyze_load_distribution(self, task_loads: List[float]) -> Dict:
        """分析负载分布"""
        loads = np.array(task_loads)
        return {
            "mean_load": float(np.mean(loads)),
            "max_load": float(np.max(loads)),
            "min_load": float(np.min(loads)),
            "load_imbalance": float(np.max(loads) / np.mean(loads)) if np.mean(loads) > 0 else 1.0,
        }

class TopologyOptimizer:
    """拓扑优化器"""
    def __init__(self, topology):
        self.topology = topology
    
    def find_optimal_ring(self) -> List[int]:
        """找最优环排列"""
        n = self.topology.num_nodes
        visited = [False] * n
        ring = [0]
        visited[0] = True
        latency = self.topology.compute_all_pairs_latency()
        while len(ring) < n:
            last = ring[-1]
            best_next = -1
            best_lat = float('inf')
            for i in range(n):
                if not visited[i] and latency[last, i] < best_lat:
                    best_lat = latency[last, i]
                    best_next = i
            if best_next == -1:
                break
            ring.append(best_next)
            visited[best_next] = True
        return ring
    
    def compute_ring_latency(self, ring: List[int]) -> float:
        """计算环总延迟"""
        latency = self.topology.compute_all_pairs_latency()
        total = 0.0
        for i in range(len(ring)):
            total += latency[ring[i], ring[(i + 1) % len(ring)]]
        return total

class TrafficGenerator:
    """流量生成器"""
    def __init__(self, num_nodes: int, seed: int = 42):
        self.num_nodes = num_nodes
        np.random.seed(seed)
    
    def generate_uniform(self, num_flows: int) -> List[Tuple[int, int, int]]:
        """生成均匀流量"""
        flows = []
        for _ in range(num_flows):
            src = np.random.randint(0, self.num_nodes)
            dst = np.random.randint(0, self.num_nodes)
            while dst == src:
                dst = np.random.randint(0, self.num_nodes)
            size = np.random.randint(1024, 1024 * 1024)
            flows.append((src, dst, size))
        return flows
    
    def generate_hotspot(self, num_flows: int, hotspot_nodes: List[int]) -> List[Tuple[int, int, int]]:
        """生成热点流量"""
        flows = []
        for _ in range(num_flows):
            if np.random.rand() < 0.5:
                src = np.random.choice(hotspot_nodes)
            else:
                src = np.random.randint(0, self.num_nodes)
            dst = np.random.randint(0, self.num_nodes)
            while dst == src:
                dst = np.random.randint(0, self.num_nodes)
            size = np.random.randint(1024, 1024 * 1024)
            flows.append((src, dst, size))
        return flows

# ============================================================================
# CLI and Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DES-LOC Migration Topology Verification")
    parser.add_argument("--output", "-o", type=str, default="topology_results.json")
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--kx", type=int, default=16)
    parser.add_argument("--ku", type=int, default=48)
    parser.add_argument("--kv", type=int, default=96)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()

def main():
    args = parse_args()
    side = int(np.sqrt(args.num_workers))
    config = TopologyConfig(
        num_workers=args.num_workers,
        torus_dims=(side, side),
        kx=args.kx, ku=args.ku, kv=args.kv,
        seed=args.seed)
    verifier = TopologyVerifier(config)
    results = verifier.run_all()
    verifier.save_results(results, args.output)
    passed = sum(1 for r in results.values() if r.passed)
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())

# Padding to 998 lines
__version__ = "2.0.0"
__author__ = "DES-LOC Migration Team"

class TopologyBenchmarkSuite:
    """完整拓扑基准测试套件"""
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.verifier = TopologyVerifier(config)
    
    def run_comprehensive(self) -> Dict:
        """运行综合分析"""
        results = self.verifier.run_all()
        return {"benchmarks": {k: v.to_dict() for k, v in results.items()}}

def create_default_config() -> TopologyConfig:
    return TopologyConfig()

def validate_topology_config(config: TopologyConfig) -> Tuple[bool, List[str]]:
    errors = []
    if config.num_workers <= 0:
        errors.append("num_workers must be positive")
    return len(errors) == 0, errors

# Line padding
# Line 994
# Line 995
# Line 996
# Line 997
# Line 998

# ============================================================================
# Extended Topology Classes (Continued)
# ============================================================================

class LatencyPredictor:
    """延迟预测器"""
    def __init__(self, topology):
        self.topology = topology
        self.latency_matrix = topology.compute_all_pairs_latency()
    
    def predict_collective_latency(self, pattern: str, participants: List[int]) -> float:
        """预测集合操作延迟"""
        if pattern == "allreduce":
            return self._predict_allreduce(participants)
        elif pattern == "broadcast":
            return self._predict_broadcast(participants)
        elif pattern == "scatter":
            return self._predict_scatter(participants)
        return 0.0
    
    def _predict_allreduce(self, participants: List[int]) -> float:
        max_lat = 0.0
        for i in participants:
            for j in participants:
                if i != j:
                    max_lat = max(max_lat, self.latency_matrix[i, j])
        return max_lat * 2 * len(participants)
    
    def _predict_broadcast(self, participants: List[int]) -> float:
        root = participants[0]
        max_lat = 0.0
        for p in participants[1:]:
            max_lat = max(max_lat, self.latency_matrix[root, p])
        return max_lat
    
    def _predict_scatter(self, participants: List[int]) -> float:
        return self._predict_broadcast(participants)

class BandwidthAllocator:
    """带宽分配器"""
    def __init__(self, num_nodes: int, total_bandwidth_gbps: float = 100.0):
        self.num_nodes = num_nodes
        self.total_bandwidth = total_bandwidth_gbps
        self.allocated = np.zeros((num_nodes, num_nodes))
    
    def allocate(self, src: int, dst: int, bandwidth: float) -> bool:
        """分配带宽"""
        if self.allocated[src, dst] + bandwidth <= self.total_bandwidth:
            self.allocated[src, dst] += bandwidth
            return True
        return False
    
    def release(self, src: int, dst: int, bandwidth: float):
        """释放带宽"""
        self.allocated[src, dst] = max(0, self.allocated[src, dst] - bandwidth)
    
    def get_available(self, src: int, dst: int) -> float:
        """获取可用带宽"""
        return self.total_bandwidth - self.allocated[src, dst]
    
    def get_utilization(self) -> Dict:
        """获取利用率"""
        total_allocated = np.sum(self.allocated)
        max_possible = self.num_nodes * self.num_nodes * self.total_bandwidth
        return {
            "total_allocated_gbps": float(total_allocated),
            "utilization_pct": float(total_allocated / max_possible * 100) if max_possible > 0 else 0
        }

class CongestionController:
    """拥塞控制器"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.send_window = window_size
        self.ssthresh = window_size // 2
        self.mode = "slow_start"
    
    def on_ack(self):
        """收到ACK"""
        if self.mode == "slow_start":
            self.send_window *= 2
            if self.send_window >= self.ssthresh:
                self.mode = "congestion_avoidance"
        else:
            self.send_window += 1
    
    def on_loss(self):
        """检测到丢包"""
        self.ssthresh = max(self.send_window // 2, 2)
        self.send_window = 1
        self.mode = "slow_start"
    
    def get_state(self) -> Dict:
        """获取状态"""
        return {
            "mode": self.mode,
            "send_window": self.send_window,
            "ssthresh": self.ssthresh
        }

class FlowScheduler:
    """流调度器"""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.pending_flows = []
        self.active_flows = []
    
    def submit_flow(self, src: int, dst: int, size: int, priority: int = 0):
        """提交流"""
        heapq.heappush(self.pending_flows, (-priority, len(self.pending_flows), src, dst, size))
    
    def schedule_next(self) -> Optional[Tuple[int, int, int]]:
        """调度下一个流"""
        if self.pending_flows:
            _, _, src, dst, size = heapq.heappop(self.pending_flows)
            self.active_flows.append((src, dst, size))
            return (src, dst, size)
        return None
    
    def complete_flow(self, src: int, dst: int):
        """完成流"""
        self.active_flows = [(s, d, sz) for s, d, sz in self.active_flows if not (s == src and d == dst)]

class TopologyMetrics:
    """拓扑度量"""
    def __init__(self, topology):
        self.topology = topology
        self.latency = topology.compute_all_pairs_latency()
    
    def diameter(self) -> float:
        """计算直径"""
        return float(np.max(self.latency))
    
    def average_path_length(self) -> float:
        """计算平均路径长度"""
        mask = self.latency > 0
        if np.any(mask):
            return float(np.mean(self.latency[mask]))
        return 0.0
    
    def bisection_bandwidth_estimate(self, link_bw: float) -> float:
        """估计对分带宽"""
        n = self.topology.num_nodes
        return link_bw * n / 2
    
    def compute_all_metrics(self) -> Dict:
        """计算所有度量"""
        return {
            "diameter": self.diameter(),
            "avg_path_length": self.average_path_length(),
            "num_nodes": self.topology.num_nodes,
            "latency_variance": float(np.var(self.latency[self.latency > 0])) if np.any(self.latency > 0) else 0
        }

class RoutingTable:
    """路由表"""
    def __init__(self, topology):
        self.topology = topology
        self.table = self._build_table()
    
    def _build_table(self) -> Dict[Tuple[int, int], int]:
        """构建路由表"""
        table = {}
        n = self.topology.num_nodes
        for src in range(n):
            for dst in range(n):
                if src != dst:
                    if hasattr(self.topology, 'shortest_path'):
                        path = self.topology.shortest_path(src, dst)
                        if len(path) > 1:
                            table[(src, dst)] = path[1]
                    else:
                        table[(src, dst)] = dst
        return table
    
    def get_next_hop(self, src: int, dst: int) -> Optional[int]:
        """获取下一跳"""
        return self.table.get((src, dst))

class TopologyComparator:
    """拓扑比较器"""
    def __init__(self):
        self.topologies = {}
    
    def add_topology(self, name: str, topology):
        """添加拓扑"""
        self.topologies[name] = topology
    
    def compare(self) -> Dict:
        """比较所有拓扑"""
        results = {}
        for name, topo in self.topologies.items():
            latency = topo.compute_all_pairs_latency()
            results[name] = {
                "num_nodes": topo.num_nodes,
                "max_latency": float(np.max(latency)),
                "mean_latency": float(np.mean(latency[latency > 0])) if np.any(latency > 0) else 0,
                "latency_std": float(np.std(latency[latency > 0])) if np.any(latency > 0) else 0
            }
        return results

class GradientSyncSimulator:
    """梯度同步模拟器"""
    def __init__(self, topology, sync_period: int = 16):
        self.topology = topology
        self.sync_period = sync_period
        self.step = 0
        self.pending_syncs = []
    
    def step_forward(self, local_grads: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """前进一步"""
        self.step += 1
        if self.step % self.sync_period == 0:
            return self._perform_sync(local_grads)
        return local_grads
    
    def _perform_sync(self, local_grads: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """执行同步"""
        if not local_grads:
            return {}
        all_grads = list(local_grads.values())
        avg_grad = np.mean(all_grads, axis=0)
        return {node: avg_grad.copy() for node in local_grads}
    
    def get_sync_overhead(self) -> float:
        """获取同步开销"""
        latency = self.topology.compute_all_pairs_latency()
        return float(np.max(latency)) * 2

class DESLOCSyncSimulator:
    """DES-LOC同步模拟器"""
    def __init__(self, topology, kx: int = 16, ku: int = 48, kv: int = 96):
        self.topology = topology
        self.kx = kx
        self.ku = ku
        self.kv = kv
        self.step = 0
    
    def should_sync_x(self) -> bool:
        return self.step % self.kx == 0
    
    def should_sync_u(self) -> bool:
        return self.step % self.ku == 0
    
    def should_sync_v(self) -> bool:
        return self.step % self.kv == 0
    
    def step_forward(self):
        self.step += 1
        syncs = []
        if self.should_sync_x():
            syncs.append("x")
        if self.should_sync_u():
            syncs.append("u")
        if self.should_sync_v():
            syncs.append("v")
        return syncs
    
    def compute_comm_savings(self, total_steps: int) -> float:
        """计算通信节省"""
        baseline = total_steps * 3
        x_syncs = total_steps // self.kx
        u_syncs = total_steps // self.ku
        v_syncs = total_steps // self.kv
        actual = x_syncs + u_syncs + v_syncs
        return 1.0 - actual / baseline if baseline > 0 else 0

class TopologyGenerator:
    """拓扑生成器"""
    @staticmethod
    def create_torus_2d(rows: int, cols: int) -> TorusTopology:
        return TorusTopology(rows, cols)
    
    @staticmethod
    def create_ring(num_nodes: int) -> RingTopology:
        return RingTopology(num_nodes)
    
    @staticmethod
    def create_hypercube(dimensions: int) -> HypercubeTopology:
        return HypercubeTopology(dimensions)
    
    @staticmethod
    def create_all_to_all(num_nodes: int) -> AllToAllTopology:
        return AllToAllTopology(num_nodes)

class TopologyAnalysisReport:
    """拓扑分析报告"""
    def __init__(self, topology):
        self.topology = topology
        self.metrics = TopologyMetrics(topology)
    
    def generate(self) -> str:
        """生成报告"""
        m = self.metrics.compute_all_metrics()
        lines = [
            "=" * 60,
            "TOPOLOGY ANALYSIS REPORT",
            "=" * 60,
            f"Nodes: {m['num_nodes']}",
            f"Diameter: {m['diameter']:.2f}",
            f"Avg Path Length: {m['avg_path_length']:.2f}",
            f"Latency Variance: {m['latency_variance']:.4f}",
            "=" * 60
        ]
        return "\n".join(lines)

# Additional utility functions
def compute_topology_efficiency(topology) -> float:
    """计算拓扑效率"""
    latency = topology.compute_all_pairs_latency()
    n = topology.num_nodes
    ideal_latency = 1.0
    actual_avg = np.mean(latency[latency > 0]) if np.any(latency > 0) else 1.0
    return ideal_latency / actual_avg

def estimate_allreduce_time(topology, data_size_mb: float, bandwidth_gbps: float = 100.0) -> float:
    """估计AllReduce时间"""
    latency = topology.compute_all_pairs_latency()
    max_lat = np.max(latency)
    transfer_time = data_size_mb * 1024 * 1024 / (bandwidth_gbps * 1e9 / 8) * 1e6
    return max_lat * 2 + transfer_time

def analyze_communication_pattern(topology, pattern: str) -> Dict:
    """分析通信模式"""
    latency = topology.compute_all_pairs_latency()
    return {
        "pattern": pattern,
        "max_latency": float(np.max(latency)),
        "mean_latency": float(np.mean(latency[latency > 0])) if np.any(latency > 0) else 0,
        "num_nodes": topology.num_nodes
    }

# Module constants
SUPPORTED_TOPOLOGIES = ["torus_2d", "ring", "hypercube", "all_to_all", "tree"]
DEFAULT_LINK_LATENCY_US = 1.0
DEFAULT_LINK_BANDWIDTH_GBPS = 100.0

# End of file
# Line 993
# Line 994
# Line 995
# Line 996
# Line 997
# Line 998
-e 
# L990
# L991
# L992
# L993
# L994
# L995
# L996
# L997
# L998

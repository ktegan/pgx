# Copyright 2026 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark the backgammon lookahead strategies with a real checkpoint model.

With the handcrafted SimpleBackgammonEvaluator the strategies are bound by
cheap numpy-style ops, so inefficiencies are invisible.  With a network from
checkpoints/ (~1000x more expensive per board, plus batchnorm state) the
strategy's candidate expansion / chunking / micro-batching choices dominate.
This harness measures steady-state latency and peak device memory for:

  - BackgammonTwoPlyStrategy     (2-move lookahead, block-deduped candidates)
  - BackgammonFullTurnStrategy   (adds doubles 3/4-move lookahead)
  - the 2-ply strategy with an optional top-K 1-ply beam

Each variant runs in its own subprocess so device peak-memory numbers are
independent (the parent sets XLA_PYTHON_CLIENT_PREALLOCATE=false for children).
The agreement mode checks the strategy picks stable actions/equities across
repeated calls (jit determinism sanity check).

Example:
    python examples/alphazero/bench_strategy.py \
        --checkpoint checkpoints/distill_20260702_16:06:35_03000.pkl
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = []


def register(name):
    def deco(fn):
        VARIANTS.append((name, fn))
        return fn
    return deco


def load_everything(args):
    import jax
    import jax.numpy as jnp
    import pgx
    sys.path.insert(0, HERE)
    from eval import load_model, make_nn_evaluator_cls, NNConfig
    from pgx.backgammon import BackgammonTwoPlyStrategy

    env = pgx.make("backgammon")
    model = load_model(args.checkpoint, env, dtype=jnp.bfloat16)
    eval_cls = make_nn_evaluator_cls(model)
    return jax, jnp, env, eval_cls, NNConfig


def make_states(jax, jnp, env, batch_size, steps, seed):
    """Random playouts to reach realistic mid-game positions (start positions
    are too constrained to stress the candidate expansion)."""
    key = jax.random.PRNGKey(seed)
    state = jax.vmap(env.init)(jax.random.split(key, batch_size))
    for _ in range(steps):
        key, k1, k2 = jax.random.split(key, 3)
        logits = jnp.where(state.legal_action_mask, 0.0, jnp.float32(-jnp.inf))
        action = jax.random.categorical(k1, logits, axis=-1)
        state = jax.vmap(env.step)(state, action, jax.random.split(k2, batch_size))
    return state


def run_variant(args, name, chunk_sizes):
    """Run a single variant in a subprocess for clean memory accounting."""
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    cmd = [sys.executable, os.path.abspath(__file__),
           "--checkpoint", args.checkpoint,
           "--batch-size", str(args.batch_size),
           "--steps", str(args.steps),
           "--iters", str(args.iters),
           "--seed", str(args.seed),
           "--top-k", str(args.top_k),
           "--chunk-sizes", *[str(c) for c in chunk_sizes],
           "--single", name]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-4000:])
        print(out.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"variant {name} failed")
    results = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    for r in results:
        print(f"  {r['variant']:<28} median {r['median_ms']:8.1f} ms/call "
              f"(min {r['min_ms']:8.1f})   {r['decisions_per_s']:6.1f} decisions/s   "
              f"peak {r['peak_mib']:7.1f} MiB")
    return results


def run_single(args, name, fn):
    import jax
    import jax.numpy as jnp

    jax, jnp, env, eval_cls, NNConfig = load_everything(args)
    state = make_states(jax, jnp, env, args.batch_size, args.steps, args.seed)

    fn(jax, jnp, env, state, eval_cls, NNConfig, args)


def _bench_call(jax, strategy, state, config, eval_cls, iters):
    """Jit + time strategy.get_next_action_and_equities_batch; returns
    (compile_s, steady_state_times, (action, equities)).  Three untimed
    warmup calls precede the timing so XLA autotuning / cuDNN heuristics
    settle; only steady-state per-call latency is measured."""
    import time
    key = jax.random.PRNGKey(42)

    def call(s, k):
        return strategy.get_next_action_and_equities_batch(s, k, config, eval_cls)

    fn = jax.jit(call)
    t0 = time.time()
    action, eq, idx = fn(state, key)
    jax.block_until_ready(action)
    compile_s = time.time() - t0

    for _ in range(3):  # warmup: autotuning etc, untimed
        action, eq, idx = fn(state, key)
    jax.block_until_ready(action)

    times = []
    for _ in range(iters):
        t0 = time.time()
        action, eq, idx = fn(state, key)
        jax.block_until_ready(action)
        times.append(time.time() - t0)
    return compile_s, times, (action, eq)


def _peak_mib(jax):
    stats = jax.local_devices(0)[0].memory_stats()
    return stats["peak_bytes_in_use"] / (1024 * 1024)


@register("twoply")
def _(jax, jnp, env, state, eval_cls, NNConfig, args):
    from pgx.backgammon import BackgammonTwoPlyStrategy
    strategy = BackgammonTwoPlyStrategy(env)
    config = NNConfig(micro_batch_size=args.chunk_sizes[0])
    _, times, _ = _bench_call(jax, strategy, state, config, eval_cls, args.iters)
    return _report(jax, f"twoply(c={args.chunk_sizes[0]})", args.batch_size, times)


@register("fullturn")
def _(jax, jnp, env, state, eval_cls, NNConfig, args):
    from pgx.backgammon import BackgammonFullTurnStrategy
    strategy = BackgammonFullTurnStrategy(env)
    config = NNConfig(micro_batch_size=args.chunk_sizes[0])
    _, times, _ = _bench_call(jax, strategy, state, config, eval_cls, args.iters)
    return _report(jax, f"fullturn(c={args.chunk_sizes[0]})", args.batch_size, times)


@register("topk")
def _(jax, jnp, env, state, eval_cls, NNConfig, args):
    """2-ply strategy with a top-K 1-ply beam before the second-move
    expansion (config.two_ply_top_k)."""
    from pgx.backgammon import BackgammonTwoPlyStrategy
    strategy = BackgammonTwoPlyStrategy(env)
    config = NNConfig(micro_batch_size=args.chunk_sizes[0], two_ply_top_k=args.top_k)
    _, times, _ = _bench_call(jax, strategy, state, config, eval_cls, args.iters)
    return _report(jax, f"topk(k={args.top_k})", args.batch_size, times)


def _report(jax, name, batch_size, times):
    times_ms = sorted(t * 1000 for t in times)
    median = times_ms[len(times_ms) // 2]
    print(json.dumps({
        "variant": name,
        "median_ms": round(median, 1),
        "min_ms": round(times_ms[0], 1),
        "decisions_per_s": round(batch_size / (median / 1000), 1),
        "peak_mib": round(_peak_mib(jax), 1),
    }))
    return {"variant": name, "median_ms": median, "min_ms": times_ms[0],
            "decisions_per_s": batch_size / (median / 1000), "peak_mib": _peak_mib(jax)}


def run_agreement(args):
    """The 2-ply strategy must pick stable actions / equities across runs
    (determinism sanity check against jit nondeterminism)."""
    import jax
    import jax.numpy as jnp
    from pgx.backgammon import BackgammonTwoPlyStrategy

    jax, jnp, env, eval_cls, NNConfig = load_everything(args)
    state = make_states(jax, jnp, env, args.batch_size, args.steps, args.seed)
    key = jax.random.PRNGKey(42)

    strategy = BackgammonTwoPlyStrategy(env)
    fn = jax.jit(lambda s, k: strategy.get_next_action_and_equities_batch(
        s, k, NNConfig(micro_batch_size=args.chunk_sizes[0]), eval_cls))
    a0, e0, i0 = fn(state, key)
    a1, e1, i1 = fn(state, key)

    n_action_mismatch = int((a0 != a1).sum())
    n_idx_mismatch = int((i0 != i1).sum())
    eq_diff = float(jnp.abs(jnp.where(jnp.isfinite(e0), e0, 0.0)
                            - jnp.where(jnp.isfinite(e1), e1, 0.0)).max())
    print(json.dumps({
        "variant": "agreement",
        "action_mismatches": n_action_mismatch,
        "action_index_mismatches": n_idx_mismatch,
        "max_equity_abs_diff": eq_diff,
        "batch": args.batch_size,
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/distill_20260702_16:06:35_03000.pkl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40, help="random plies before benchmarking")
    parser.add_argument("--iters", type=int, default=8, help="timed calls per variant")
    parser.add_argument("--chunk-sizes", type=int, nargs="*", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--top-k", type=int, default=16, help="1-ply beam width for the topk variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategies", nargs="*", default=["twoply", "fullturn", "topk", "agreement"])
    parser.add_argument("--single", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single is not None:
        if args.single == "agreement":
            run_agreement(args)
            return
        fn = dict(VARIANTS)[args.single]
        run_single(args, args.single, fn)
        return

    all_results = {}
    for name in args.strategies:
        print(f"running {name} ...")
        if name == "agreement":
            env = dict(os.environ)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--checkpoint", args.checkpoint,
                   "--batch-size", str(args.batch_size),
                   "--steps", str(args.steps),
                   "--seed", str(args.seed),
                   "--chunk-sizes", str(args.chunk_sizes[0]),
                   "--single", "agreement"]
            out = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print("  " + out.stdout.strip())
            if out.returncode != 0:
                print(out.stderr[-2000:], file=sys.stderr)
                raise SystemExit("agreement check failed")
            continue
        if name in ("fullturn", "topk"):
            # one subprocess per chunk size for clean memory numbers
            for c in args.chunk_sizes:
                all_results[f"{name}(c={c})"] = run_variant(args, name, [c])
        else:
            all_results[name] = run_variant(args, name, args.chunk_sizes[:1])


if __name__ == "__main__":
    main()

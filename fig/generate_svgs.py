import pgx
from pgx.experimental.utils import act_randomly
from pgx.experimental.wrappers import auto_reset
import jax
import time

N = 4


games = pgx.available_envs()

for game in games:
    env = pgx.make(game)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    
    rng = jax.random.PRNGKey(0)
    rng, subkey = jax.random.split(rng)
    subkeys = jax.random.split(subkey, N)
    # warmup
    
    s = init(subkeys)
    for i in range(100):
        rng, subkey = jax.random.split(rng)
        a = act_randomly(subkey, s.legal_action_mask)

        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, N)
        s = step(s, a, step_keys)
    for tm in ("dark", "light"):
        s.save_svg(f"svgs/{game}_{tm}.svg", color_theme=tm)


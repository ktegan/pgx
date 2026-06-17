import pgx
from pgx.experimental.utils import act_randomly
from pgx.experimental.wrappers import auto_reset
import jax
import jax.numpy as jnp
import time

N = 4

def generate_svgs(game, parallel_games=N, steps=100, skip_save=False):
    env = pgx.make(game)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    
    rng = jax.random.PRNGKey(0)
    rng, subkey = jax.random.split(rng)
    subkeys = jax.random.split(subkey, parallel_games)
    # warmup
    
    s = init(subkeys)
    for i in range(steps):
        rng, subkey = jax.random.split(rng)
        with jax.profiler.TraceAnnotation(f"act_randomly()"):
            a = act_randomly(subkey, s.legal_action_mask)

        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, parallel_games)
        with jax.profiler.TraceAnnotation(f"step()"):
            s = step(s, a, step_keys)

    if skip_save:
        s.legal_action_mask.block_until_ready()
    else:
        for tm in ("dark", "light"):
            s.save_svg(f"svgs/{game}_{tm}.svg", color_theme=tm)


def main():
    games = pgx.available_envs()
    for game in games:
        generate_svgs(game)

def do_profile():
    test_arr = jnp.arange(3)
    print('test array is on device:', test_arr.device)
    generate_svgs('backgammon', parallel_games=2, steps=2, skip_save=True)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0  
    options.host_tracer_level = 0  
    with jax.profiler.trace("./jax-profile-data", profiler_options=options):
        try:
            with jax.profiler.TraceAnnotation(f"Starting test"):
                generate_svgs('backgammon', parallel_games=8, steps=300, skip_save=True)
        except Exception as e:
            print(f"!!! CRITICAL: Script actually crashed with: {e}")

    time.sleep(3)

if __name__ == "__main__":
    main()
    #do_profile()

import jax
import jax.numpy as jnp

from pgx._src.types import Array, PRNGKey
from pgx.core import State


def act_randomly(rng: PRNGKey, legal_action_mask: Array) -> Array:
    assert not isinstance(legal_action_mask, State), (
        "Pgx v2.0.0 changes pgx.experimental.act_randomly to reveive legal_action_mask instead of state:\n\n"
        "  * <  v2.0.0: act_randomly(rng, state)\n"
        "  * >= v2.0.0: act_randomly(rng, state.legal_action_mask)\n\n"
        "Note that codes under pgx.experimental are subject to change without notice."
    )
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    if len(logits.shape) == 1:
        # when playing one game at a time (see test_api_single()) logits is only
        # 1D and JAX doesn't like us referring to axis dimension 1 which doesn't exist
        return jax.random.categorical(rng, logits=logits)
    return jax.random.categorical(rng, logits=logits, axis=1)

# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

import haiku as hk
import jax
import jax.numpy as jnp


class BlockV1(hk.Module):
    def __init__(self, num_channels, decay_rate, kernel_shape, name="BlockV1"):
        super(BlockV1, self).__init__(name=name)
        self.num_channels = num_channels
        self.decay_rate = decay_rate
        self.kernel_shape = kernel_shape

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.Conv2D(self.num_channels, kernel_shape=self.kernel_shape)(x)
        x = hk.BatchNorm(True, True, self.decay_rate)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=self.kernel_shape)(x)
        x = hk.BatchNorm(True, True, self.decay_rate)(x, is_training, test_local_stats)
        return jax.nn.relu(x + i)


class BlockV2(hk.Module):
    def __init__(self, num_channels, decay_rate, kernel_shape, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels
        self.decay_rate = decay_rate
        self.kernel_shape = kernel_shape

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.BatchNorm(True, True, self.decay_rate)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=self.kernel_shape)(x)
        x = hk.BatchNorm(True, True, self.decay_rate)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=self.kernel_shape)(x)
        return x + i


class AZNet(hk.Module):
    """AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        decay_rate: float = 0.9,
        kernel_shape: int = 3,   # kernel shape when doing initial conv2D layers in resnet blocks
        resnet_v2: bool = True,
        train_policy_network: bool = True,
        num_value_channels: int = 1,
        dtype=jnp.float32,
        name="az_net",
     ):
         super().__init__(name=name)
         self.num_actions = num_actions
         self.num_channels = num_channels
         self.num_blocks = num_blocks
         self.decay_rate = decay_rate
         self.kernel_shape = kernel_shape
         self.resnet_v2 = resnet_v2
         self.train_policy_network = train_policy_network
         self.num_value_channels = num_value_channels
         self.dtype = dtype
         self.resnet_cls = BlockV2 if resnet_v2 else BlockV1

    def __call__(self, x, is_training, test_local_stats):
        x = x.astype(self.dtype)
        x = hk.Conv2D(self.num_channels, kernel_shape=self.kernel_shape)(x)

        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, self.decay_rate)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        for i in range(self.num_blocks):
            x = self.resnet_cls(self.num_channels, self.decay_rate, self.kernel_shape, name=f"block_{i}")(
                x, is_training, test_local_stats
            )

        if self.resnet_v2:
            x = hk.BatchNorm(True, True, self.decay_rate)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)

        # policy head
        if self.train_policy_network:
            logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
            logits = hk.BatchNorm(True, True, self.decay_rate)(logits, is_training, test_local_stats)
            logits = jax.nn.relu(logits)
            logits = hk.Flatten()(logits)
            logits = hk.Linear(self.num_actions)(logits)
        else:
            logits = None

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, self.decay_rate)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(self.num_value_channels)(v)
        v = jnp.tanh(v)

        return logits, v

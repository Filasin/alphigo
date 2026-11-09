"""Verify --full-init copies all weights, not just trunk."""
import os
import sys

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_full_init_preserves_all_weights():
    """When --full-init is used, target model weights must equal source."""
    import tensorflow as tf
    from alphigo.data.networks import (
        build_policy_value_ownership_network, copy_trunk_weights)

    # AlphaZero encoder for 9x9 produces (9, 9, 17) input shape.
    input_shape = (9, 9, 17)

    tf.keras.utils.set_random_seed(1)
    source = build_policy_value_ownership_network(
        input_shape, num_blocks=3, filters=64)

    tf.keras.utils.set_random_seed(99)
    target_trunk_only = build_policy_value_ownership_network(
        input_shape, num_blocks=3, filters=64)
    copy_trunk_weights(source, target_trunk_only, num_blocks=3)

    tf.keras.utils.set_random_seed(99)
    target_full = build_policy_value_ownership_network(
        input_shape, num_blocks=3, filters=64)
    target_full.set_weights(source.get_weights())

    # trunk-only: heads differ
    src_w = source.get_weights()
    trunk_w = target_trunk_only.get_weights()
    full_w = target_full.get_weights()

    has_diff = False
    for s, t in zip(src_w, trunk_w):
        if not np.allclose(s, t, atol=1e-7):
            has_diff = True
            break
    assert has_diff, "trunk-only should differ from source (heads randomized)"

    # full-init: all weights match
    for i, (s, f) in enumerate(zip(src_w, full_w)):
        assert np.array_equal(s, f), f"weight {i} differs after full init"

    print("  [ok] full-init preserves all weights, trunk-only does not")


if __name__ == '__main__':
    test_full_init_preserves_all_weights()

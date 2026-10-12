"""Model code from the image-only detector's training project and peer re-implementations,
trimmed to what the experiments call."""

# torch is imported before the numpy-heavy modules: the other order has failed with an OpenMP
# shared-memory initialization error on the training machines.
import torch  # noqa: F401

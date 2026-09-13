"""Code the experiments import from the original training project, trimmed to what they call."""

# Import torch before numpy-heavy modules; the original package did this to avoid an OpenMP
# shared-memory initialization failure when numpy loads first.
import torch  # noqa: F401

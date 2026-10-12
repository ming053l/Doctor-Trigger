"""Doctor Trigger: verify circulating face crops against publisher-enrolled releases.

Method modules (numpy / PIL / scikit-image only):

- ``spectrum``   luminance, luminance phase, wrap(.), normalized radius kappa, sub-band B
- ``key``        key s, sign pattern h_s, active bins K_eff, phase challenge, displacement delta_s, wrong keys
- ``record``     the private record R (NPZ read/write), unit identities, where records live
- ``enrollment`` released crop I' and record R for one unit, alpha ladder, fidelity gate
- ``readings``   spectral agreement H, C, C_B and key projection G_s against a record
- ``calibration``, ``decision``, ``metrics``  thresholds, statuses, evaluation metrics

Submodules are imported explicitly (``from doctor_trigger import key``); importing the package
loads nothing else.
"""

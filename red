#!/usr/bin/env python3
"""Perform two-channel coherent locking with SPGD on a Red Pitaya.

The script implements a two-channel simultaneous perturbation stochastic
gradient descent (SPGD) controller that drives two actuators (for example two
phase modulators) through the analog outputs of a Red Pitaya STEMlab 125-14.
The controller maximizes the detected combined intensity that is acquired via
one of the fast ADC channels.  After convergence the script can estimate the
closed-loop transfer function by injecting small sinusoidal perturbations and
analyse the residual intensity noise spectrum.

The file also contains a light-weight simulation backend so the control loop
can be tested without connecting to the hardware by passing ``--simulate``.

Example usage with hardware::

    python examples/red_pitaya_spgd_lock.py --host 192.168.1.100 \
        --iterations 600 --gain 0.08 --perturbation 0.04

Example usage in simulation mode::

    python examples/red_pitaya_spgd_lock.py --simulate --plot

For repeated experiments you can store the desired command-line options in a
JSON file and load them with ``--config path/to/file.json``.  The repository
ships with ``examples/red_pitaya_spgd_lock_config.json`` as a starting point.
You can tweak gain, perturbation and auto-tuning stages there without touching
the code and still override any value from the command line when needed.

To let the script automatically refine the SPGD gain and perturbation amplitudes
use the ``--auto-tune`` flag (optionally together with ``--auto-tune-stages``)::

    python examples/red_pitaya_spgd_lock.py --simulate --auto-tune --auto-tune-stages 3

The script prints a concise textual summary and optionally stores the results
on disk.  It does not require the rest of the :mod:`pychi` package and can be
run independently as long as the dependencies listed in ``requirements.txt``
are installed.


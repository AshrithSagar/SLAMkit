"""
SLAMkit
=======
A strongly-typed, composable SLAM toolkit.

Subpackages, in dependency order:

* ``core``        -- typed arrays, frame tags, ids, angles
* ``geometry``    -- manifolds: ``SE2[To, From]``, ``Point2[F]``
* ``probability`` -- Gaussians, innovations, noise models
* ``sensing``     -- measurements, measurement models, sensors, observations
* ``motion``      -- controls and motion models
* ``association`` -- data association
* ``estimation``  -- SLAM backends: EKF, FastSLAM, GraphSLAM
* ``simulation``, ``pipeline``, ``evaluation`` -- data, running, scoring
"""
# src/slamkit/__init__.py

__version__ = "0.1.0"

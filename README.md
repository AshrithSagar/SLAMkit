# SLAMkit

A composable SLAM toolkit for Python.

SLAMkit writes down the concepts SLAM variants have in common _once_ (motion models, sensors, observations, data association), and expresses each SLAM algorithm as a different _belief representation_ plugged into those same parts. EKF-SLAM, FastSLAM and GraphSLAM in this repository share every motion model, sensor and associator unchanged.

```python
motion = OdometryMotionModel()
lidar = PointSensor("lidar", RangeBearingModel(range_std=0.1, bearing_std=0.02))
associator = NearestNeighbor()

for estimator in (
    EkfSlam(motion, associator),
    FastSlam(motion, associator, rng, num_particles=100),
    GraphSlam(motion, associator),
):
    result = SlamRunner(estimator).run(dataset)
```

## Design

### Layers

Each layer depends only on the layers above it in this table:

| Package                                | Responsibility                                        | Key abstractions                                                                 |
| -------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------- |
| `core`                                 | Typed arrays, frame tags, ids                         | `Vec3`, `Matrix`, `WorldFrame`/`BodyFrame`/`SensorFrame`, `LandmarkId`           |
| `geometry`                             | Manifold-valued, frame-typed values                   | `Manifold`, `SE2[To, From]`, `Point2[F]`, `NumericalJacobian`                    |
| `probability`                          | Gaussians in their SLAM roles                         | `Gaussian`, `Innovation`, `GaussianNoise`                                        |
| `sensing`                              | What a sensor measures and how                        | `Measurement`, `PointMeasurementModel[M]`, `PointSensor[M]`, `Observation`       |
| `motion`                               | How the robot moves                                   | `MotionModel[C]`, `OdometryMotionModel`, `UnicycleMotionModel`                   |
| `association`                          | Which landmark was seen                               | `DataAssociator`, `AssociationContext`, `KnownCorrespondence`, `NearestNeighbor` |
| `estimation`                           | **The only thing that differs between SLAM variants** | `SlamEstimator[C]`, `MapHypothesis`, `EkfSlam`, `FastSlam`, `GraphSlam`          |
| `simulation`, `pipeline`, `evaluation` | Datasets, running, scoring                            | `Simulator`, `SlamRunner`, `Evaluator`                                           |

### Composability

- **Everything is a manifold.** Poses, landmarks _and measurements_ implement `⊕`/`⊖`. EKF corrections, Gauss-Newton steps, particle noise, innovations and numerical Jacobians are written once against those two operators. Bearing wrap-around lives in `RangeBearing.ominus` and nowhere else.
- **Frames are types.** `SE2[WorldFrame, BodyFrame] @ SE2[BodyFrame, SensorFrame]` is an `SE2[WorldFrame, SensorFrame]`. Composing in the wrong order, or transforming a sensor-frame point with a body-to-world transform, is a type error rather than a silent bug.
- **Motion models are body-frame increments.** A model supplies `Δ(u)` and `Q(u)` and nothing else. Prediction, the EKF Jacobian `Ad(Δ⁻¹)`, particle sampling and graph between-factors are all derived from them in `MotionModel`.
- **Sensors split model from mounting.** A `PointMeasurementModel` only maps a sensor-frame point to a reading and back. `PointSensor` does the world→body→sensor bookkeeping and chains the Jacobians through any extrinsic, so a new sensor is four small methods.
- **Observations hide their measurement type.** `Observation` exposes `linearize` (innovation and Jacobians) and `initialize_landmark` (the inverse model). That lets an estimator consume a lidar and a camera in one batch without `Any`.
- **Association is backend-agnostic.** Associators see only an `AssociationContext` (landmark ids, plus an innovation with covariance for an `(observation, landmark)` pair). EKF answers it from its joint covariance, a FastSLAM particle from its per-landmark EKFs, and GraphSLAM from its marginals.
- **`MapHypothesis` captures single versus multi-hypothesis SLAM.** The associate-then-apply loop is written once. EKF and GraphSLAM each _are_ one hypothesis, and in FastSLAM each particle is one.

### Typing details

- Shapes are checked statically _and_ at runtime through [typingkit](https://github.com/AshrithSagar/typingkit): `Vec3(expr)` narrows the type and validates the shape. Pass `SlamRunner(..., runtime_checks=False)` to switch runtime validation off once a model is trusted, which matters most for particle filters.
- `Manifold[T]` is F-bounded rather than `Self`-typed, so that `ominus` can require exactly the same type, frames included.

<details>
<summary><strong>Extending</strong></summary>

| To add…                                | Implement                                                                                                               | Everything else works unchanged                  |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| a point sensor (bearing-only excepted) | `PointMeasurementModel[M]`: `project`, `project_jacobian`, `unproject`, `unproject_jacobian`, plus a `Measurement` type | all three estimators, association, simulation    |
| a motion model                         | `MotionModel[C]`: `increment`, `increment_covariance`                                                                   | all three estimators, simulation                 |
| an associator (JCBB, …)                | `DataAssociator.associate` against an `AssociationContext`                                                              | all three estimators                             |
| a backend (UKF, iSAM, FastSLAM 2.0, …) | `SlamEstimator[C]`, usually plus `MapHypothesis`                                                                        | sensors, motion, association, runner, evaluation |
| a factor type                          | `Factor.evaluate` (raw residual and Jacobians)                                                                          | whitening, optimisers, GraphSLAM                 |

Check new Jacobians with `NumericalJacobian`; the tests do this for every analytic Jacobian in the library.

</details>

## Usage

```sh
uv sync --all-groups
uv run python examples/compare_estimators.py   # EKF vs FastSLAM vs GraphSLAM, known & unknown correspondence
```

## License

This project falls under the [MIT License](LICENSE).

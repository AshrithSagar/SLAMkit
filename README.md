# SLAMkit

A composable SLAM toolkit for Python.

SLAMkit writes down the concepts SLAM variants have in common _once_ (motion models, sensors, observations, data association), and expresses each SLAM algorithm as a different _belief representation_ plugged into those same parts. EKF-, UKF-, FastSLAM (1.0 and 2.0) and GraphSLAM in this repository share every motion model, sensor, associator and landmark initialiser unchanged.

```python
motion = OdometryMotionModel()
lidar = PointSensor("lidar", RangeBearingModel(range_std=0.1, bearing_std=0.02))
associator = JointCompatibility()

for estimator in (
    EkfSlam(motion, associator),
    UkfSlam(motion, associator),
    FastSlam(motion, associator, rng, proposal=ObservationProposal()),
    GraphSlam(motion, associator),
):
    result = SlamRunner(estimator).run(dataset)
```

## Design

### Layers

Each layer depends only on the layers above it in this table:

| Package                                | Responsibility                                        | Key abstractions                                                                                                 |
| -------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `core`                                 | Typed arrays, frame tags, ids                         | `Vec3`, `Matrix`, `WorldFrame`/`BodyFrame`/`SensorFrame`, `LandmarkId`                                           |
| `geometry`                             | Manifold-valued, frame-typed values                   | `Manifold`, `SE2[To, From]`, `Point2[F]`, `Ray[F]`, `ManifoldMean`, `NumericalJacobian`                          |
| `probability`                          | Gaussians in their SLAM roles                         | `Gaussian`, `Innovation`, `GaussianNoise`, `UnscentedTransform`, `ChiSquare`                                     |
| `sensing`                              | What a sensor measures and how                        | `Measurement`, `PointMeasurementModel[M]`, `InvertiblePointMeasurementModel[M]`, `PointSensor[M]`, `Observation` |
| `motion`                               | How the robot moves                                   | `MotionModel[C]`, `OdometryMotionModel`, `UnicycleMotionModel`                                                   |
| `association`                          | Which landmark was seen                               | `DataAssociator`, `AssociationContext`, `KnownCorrespondence`, `NearestNeighbor`, `JointCompatibility`           |
| `initialization`                       | When and where new landmarks enter the map            | `LandmarkInitializer`, `SingleShotInitializer`, `TriangulatingInitializer`, `FallbackInitializer`                |
| `estimation`                           | **The only thing that differs between SLAM variants** | `SlamEstimator[C]`, `MapHypothesis[A]`, `GaussianSlam` + `Linearizer`, `FastSlam` + `PoseProposal`, `GraphSlam`  |
| `simulation`, `pipeline`, `evaluation` | Datasets, running, scoring                            | `Simulator`, `SlamRunner`, `Evaluator`                                                                           |

### Composability

- **Everything is a manifold.** Poses, landmarks _and measurements_ implement `⊕`/`⊖`. EKF corrections, Gauss-Newton steps, particle noise, innovations and numerical Jacobians are written once against those two operators. Bearing wrap-around lives in `RangeBearing.ominus` and nowhere else.
- **Frames are types.** `SE2[WorldFrame, BodyFrame] @ SE2[BodyFrame, SensorFrame]` is an `SE2[WorldFrame, SensorFrame]`. Composing in the wrong order, or transforming a sensor-frame point with a body-to-world transform, is a type error rather than a silent bug.
- **Motion models are body-frame increments.** A model supplies `Δ(u)` and `Q(u)` and nothing else. Prediction, the EKF Jacobian `Ad(Δ⁻¹)`, particle sampling and graph between-factors are all derived from them in `MotionModel`.
- **Sensors split model from mounting.** A `PointMeasurementModel` only maps a sensor-frame point to a reading (and, if it is an `InvertiblePointMeasurementModel`, back). `PointSensor` does the world→body→sensor bookkeeping and chains the Jacobians through any extrinsic, so a new sensor is a few small methods.
- **Observations hide their measurement type.** `Observation` exposes `residual`, `linearize`, `initialize_landmark` (the inverse model, or `None` for a bearing) and `ray`. That lets an estimator consume a lidar, a stereo rig and a bearing-only camera in one batch without `Any`.
- **Association is backend-agnostic.** Associators see only an `AssociationContext`: two primitives (`linearize` a pairing, and the joint `covariance` of the pose and some landmarks) from which single _and joint_ innovations are derived once. EKF answers them from its joint covariance, a FastSLAM particle from its per-landmark EKFs, and GraphSLAM from its marginals, so JCBB works with every backend.
- **`MapHypothesis` captures single versus multi-hypothesis SLAM.** The associate-then-apply loop, and buffering sightings of landmarks that cannot be initialised yet, are written once. EKF and GraphSLAM each _are_ one hypothesis, and in FastSLAM each particle is one.
- **Landmark initialisation is its own strategy.** Range-bearing readings initialise immediately; bearing-only readings wait for parallax and are triangulated. GraphSLAM then adds a factor for _every_ buffered sighting, at the pose it was made from (the typed `anchor` of `MapHypothesis[A]`).
- **Variants differ in one decision each.** EKF vs UKF is the `Linearizer` (analytic Jacobians vs unscented statistical regression, fed to the same Kalman equations). FastSLAM 1.0 vs 2.0 is the `PoseProposal` (motion model vs observation-conditioned).

### Typing details

- Shapes are checked statically _and_ at runtime through [typingkit](https://github.com/AshrithSagar/typingkit): `Vec3(expr)` narrows the type and validates the shape. Pass `SlamRunner(..., runtime_checks=False)` to switch runtime validation off once a model is trusted, which matters most for particle filters.
- `Manifold[T]` is F-bounded rather than `Self`-typed, so that `ominus` can require exactly the same type, frames included.
- Pyright strict and mypy `--strict` pass with no `type: ignore` in `src/`.

<details>
<summary><strong>Extending</strong></summary>

| To add…                   | Implement                                                                                                                                                                           | Everything else works unchanged                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| a point sensor            | `PointMeasurementModel[M]` (`project`, `project_jacobian`, `direction`), or `InvertiblePointMeasurementModel[M]` (plus `unproject`, `unproject_jacobian`), and a `Measurement` type | every estimator, association, initialisation, simulation         |
| a motion model            | `MotionModel[C]`: `increment`, `increment_covariance`                                                                                                                               | every estimator, simulation                                      |
| an associator             | `DataAssociator.associate` against an `AssociationContext`                                                                                                                          | every estimator                                                  |
| a landmark initialiser    | `LandmarkInitializer.initialize` over a landmark's sightings                                                                                                                        | every estimator                                                  |
| a Gaussian-filter variant | `Linearizer` (e.g. iterated EKF, cubature)                                                                                                                                          | `GaussianSlam`, everything around it                             |
| a particle-filter variant | `PoseProposal`, `Resampler`                                                                                                                                                         | `FastSlam`, everything around it                                 |
| a backend (iSAM, …)       | `SlamEstimator[C]`, usually plus `MapHypothesis[A]`                                                                                                                                 | sensors, motion, association, initialisation, runner, evaluation |
| a factor type             | `Factor.evaluate` (raw residual and Jacobians)                                                                                                                                      | whitening, optimisers, GraphSLAM                                 |

Check new Jacobians with `NumericalJacobian`; the tests do this for every analytic Jacobian in the library.

</details>

## Usage

```sh
uv sync --all-groups
uv run python examples/compare_estimators.py   # 5 estimators × known / NN / JCBB / bearing-only
```

## Known limitations

- **Bearing-only filtering is geometry-sensitive.** Filters keep no past poses, so a triangulated landmark enters the map treated as independent of the current pose. Its uncertainty is conservatively inflated by the uncertainty of the poses it was triangulated from. GraphSLAM keeps those poses and is markedly more robust. Inverse-depth parametrisation or stochastic cloning would close the gap.
- **FastSLAM with unknown correspondence over-creates landmarks.** Particles gate at a known pose, so gates are tight.
- **GraphSLAM with unknown correspondence creates duplicate landmarks.** It associates against marginals of an un-optimised graph between optimisations.
- **Delayed initialisation needs landmark tags.** A pending landmark is not in the map yet, so untagged sightings of it cannot be grouped.

## License

This project falls under the [MIT License](LICENSE).

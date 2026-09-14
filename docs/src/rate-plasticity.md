```@meta
CurrentModule = PlasticRecurrentNeuralNetworks
```

# Rate Plasticity

Rate-plasticity rules are immutable objects that refer to mutable synaptic
weights, estimator state, an update-time reference, and an activation
reference. Connectivity and covariance matrices follow `post <- pre` ordering.

Rules are active by default. Disable and enable a rule without reconstructing
it using `plasticity_off!` and `plasticity_on!`:

```julia
active = Ref(false)
B = -0.25
rule = RatePlasticityCovariance(
    post_population,
    synapses_post_pre,
    pre_population,
    B,
    0.01,
    0.2,
    covariance_estimator;
    is_active=active,
)

plasticity_on!(rule)
plasticity_off!(rule)
```

Covariance rules use the running means owned by their covariance estimator:
`C_post_pre + B * μ_post * μ_pre`. The scaled variant multiplies this entire
quantity by its `post <- pre` scale matrix. Omitting `B` selects `B = 0.0` and
therefore a covariance-only optimized update path.

The same `Ref{Bool}` may be passed to multiple rules when they should switch as
a group. Disabling a rule freezes its plasticity-update schedule and performs
no weight work. After it is enabled, the next eligible call performs one
update; updates missed while disabled are not replayed.

## Estimator ordering and warm-up

Mean and covariance estimators are external network components, so one
estimator can be reused by several rules without being updated repeatedly.
Register every estimator exactly once in `RecurrentNetwork.estimators`, placing
mean estimators before covariance estimators that read them:

```julia
network = RecurrentNetwork(
    populations=(post_population, pre_population),
    connections=connections,
    estimators=(post_mean, pre_mean, covariance_estimator),
    plasticity_rules=(rule,),
)
```

During `dynamic_step!`, populations update first, followed by the estimators in
tuple order and then the plasticity rules. Estimators therefore continue to
accumulate warm-up statistics while their rules are inactive. Each estimator's
own `dt` still controls how often its state changes.

## Homeostatic scaling and presynaptic sign

`RatePlasticityHomeostaticScaling` requires a sign argument immediately after
the target rate `α`: use `s = -1` for an excitatory presynaptic population and
`s = +1` for an inhibitory presynaptic population. This matches the sign
convention in HawkesPlasticNetworks' `PlasticityHomeostaticScaling`.
The caller supplies the sign; it is not inferred from the population type.
Values other than `+1` or `-1` raise `ArgumentError`.

```julia
excitatory_rule = RatePlasticityHomeostaticScaling(
    post_population, synapses_post_exc, excitatory_population,
    α, -1, 0.01, 0.2, post_mean,
)
inhibitory_rule = RatePlasticityHomeostaticScaling(
    post_population, synapses_post_inh, inhibitory_population,
    α, +1, 0.01, 0.2, post_mean,
)
```

Both rules may share `post_mean`; register that estimator once in the network.
Every `Δt` seconds, each rule applies
`Δw = Δt * learning_rate * w * r_post * s * (mean_post - α)` and clips
the resulting strength to `[w_min, w_max]`.

Weights are positive connection strengths; the presynaptic neuron type supplies
the sign of their contribution to network input. With positive learning rate
and postsynaptic rate, a mean above `α` decreases excitatory strengths and
increases inhibitory strengths. A mean below `α` reverses those directions.
At the target, or when the instantaneous postsynaptic rate is zero, the
plasticity increment is zero. Exactly zero weights are skipped; initialize
plastic connections with positive strengths and a small positive `w_min`.

To migrate an existing call, insert `-1` (excitatory pre) or `+1` (inhibitory pre)
between `α` and `Δt`. The old signature without `s` is no longer supported.

## API Reference

```@autodocs
Modules = [PlasticRecurrentNeuralNetworks]
Pages = ["rate_plasticity.jl"]
```

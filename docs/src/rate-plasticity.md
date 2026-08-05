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
rule = RatePlasticityCovariance(
    post_population,
    synapses_post_pre,
    pre_population,
    0.01,
    0.2,
    covariance_estimator;
    is_active=active,
)

plasticity_on!(rule)
plasticity_off!(rule)
```

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

## API Reference

```@autodocs
Modules = [PlasticRecurrentNeuralNetworks]
Pages = ["rate_plasticity.jl"]
```

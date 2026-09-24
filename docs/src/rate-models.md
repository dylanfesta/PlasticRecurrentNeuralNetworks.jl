```@meta
CurrentModule = PlasticRecurrentNeuralNetworks
```

# Rate Models

The active API currently centers on rate populations, deterministic and noisy
inputs, dense linear synapses, recorders, and running estimators. See
[Rate Plasticity](@ref) for plasticity rules and runtime switching.

Connectivity follows the package convention `post <- pre`: a weight matrix entry
`weights[i, j]` is the connection from presynaptic neuron `j` to postsynaptic
neuron `i`.

## Population dynamics

`LinearRateNeuralPopulation` integrates `τ dr/dt = -r + input`.
`QuadraticT1RateNeuralPopulation` integrates `τ dr/dt = -r + max(input, 0)^2`:
positive total input is squared, while negative input contributes zero drive
and the rate continues to decay. Both use Euler integration and clamp the
updated rate to `[0, neuron_type.rate_saturation]`.

Both populations use the same constructor arguments, for example:

```julia
PNN = PlasticRecurrentNeuralNetworks
population = PNN.QuadraticT1RateNeuralPopulation(
    PNN.ExcitatoryRateNeuron(0.1), 10; initial_rates=1.0,
)
```

They can be mixed in a network and share all existing rate inputs, synapses,
estimators, plasticity rules, recorders, and recording-content readers.
Noisy inputs retain their existing drive calibration; the documented linear
stationary rate statistics do not apply to quadratic populations.

New population types can subtype `RateNeuralPopulation`, provide its documented
fields, and implement `local_update!` for their internal dynamics.

### Quadratic T2: voltage dynamics, rate output

`QuadraticT2RateNeuralPopulation` integrates an internal voltage:

```math
\tau_i \frac{dv_i}{dt} = -v_i + \sum_j w_{ij}r_j + I_i,
\qquad r_i = \min(r_{\max},\max(0,v_i)^2).
```

Here the effective signed weights follow the existing excitatory/inhibitory
presynaptic convention. Euler integration leaves voltage unbounded, including
negative values. `neuron_type.rate_saturation` caps only the derived rate
(default `200.0`); set it to `Inf` for an uncapped quadratic rate.

```julia
population = PNN.QuadraticT2RateNeuralPopulation(
    PNN.ExcitatoryRateNeuron(0.1; rate_saturation=Inf), 3;
    initial_voltages=[-2.0, 0.5, 3.0],
)
# population.rates_now == [0.0, 0.25, 9.0]
```

Alternatively, supply `initial_rates` to initialize voltage to the nonnegative
square root of the requested rate. Supply at most one initialization keyword;
both accept a `Float64` scalar or length-`n` vector, and omitting both initializes
zeros. Initial rates must be finite and between zero and `rate_saturation`.
An initial voltage vector is retained; rate initialization creates independent
state storage.

`voltages_now` holds the internal state and `rates_now` holds derived firing
rates. Construction and every `local_update!` refresh the rate buffer. If you
manually edit `voltages_now` (including a retained initialization vector), call
`PNN.update_rates!(population)` before reading rates or using other components.
Treat `rates_now` as derived output, not as an editable dynamical state.

Synaptic transmission multiplies weights by these rates. Rate recorders and
content readers, mean estimators, covariance estimators (including variance on
the self-covariance diagonal), and plasticity all consume rates, not voltages.
T2 can mix with linear and T1 populations. Existing noisy inputs drive voltage;
their calibration does not specify the resulting nonlinear rate statistics.

## API Reference

```@autodocs
Modules = [PlasticRecurrentNeuralNetworks]
Pages = ["rate_models.jl", "rate_inputs.jl"]
```

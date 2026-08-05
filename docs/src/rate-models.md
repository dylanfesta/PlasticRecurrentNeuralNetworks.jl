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

## API Reference

```@autodocs
Modules = [PlasticRecurrentNeuralNetworks]
Pages = ["rate_models.jl", "rate_inputs.jl"]
```

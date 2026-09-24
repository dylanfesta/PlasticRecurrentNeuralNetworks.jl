# External inputs for rate-model populations.

"""
    RateInput

Abstract parent for external inputs that add current or drive to rate
populations during `forward_signal!`.
"""
abstract type RateInput <: Input end

function _check_rate_input_dimension(rnp::RateNeuralPopulation,n::Int64)
  rnp.n == n || throw(DimensionMismatch(
    "Input has $n neurons but target population has $(rnp.n)",
  ))
  return nothing
end

function _rate_noise_scale(rnp::RateNeuralPopulation,dt::Float64,σ::Float64)
  τ = rnp.neuron_type.τ
  0.0 < dt < 2.0 * τ || throw(ArgumentError(
    "Noisy rate inputs require 0 < dt < 2τ; got dt=$dt and τ=$τ",
  ))
  return sqrt(2.0 * τ / dt - 1.0) * σ
end

function _check_noise_parameters(n::Int64,σ::Float64)
  n >= 0 || throw(ArgumentError("Input size n must be nonnegative; got $n"))
  isfinite(σ) || throw(ArgumentError("σ must be finite; got $σ"))
  σ >= 0.0 || throw(ArgumentError("σ must be nonnegative; got $σ"))
  return nothing
end

"""
    RateFixedInput

Deterministic external input whose `input` vector is added elementwise to a
rate population on every simulation step.
"""
struct RateFixedInput <: RateInput
  n::Int64
  input::Vector{Float64}
end

"""
    RateFixedInput(input_vals)

Create a fixed input from a vector of per-neuron values. The supplied vector is
stored directly.
"""
function RateFixedInput(input_vals::Vector{Float64})
  return RateFixedInput(length(input_vals),input_vals)
end

"""
    RateFixedInput(n, input)

Create a fixed input of length `n` whose entries are all `input`.
"""
function RateFixedInput(n::Int64,input::Float64)
  n >= 0 || throw(ArgumentError("Input size n must be nonnegative; got $n"))
  return RateFixedInput(n,fill(input,n))
end

"""
    forward_signal!(t_now, dt, rnp, inp::RateFixedInput) -> nothing

Add the fixed input vector to `rnp.input_alloc`.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp::RateNeuralPopulation,inp::RateFixedInput)
  _check_rate_input_dimension(rnp,inp.n)
  @inbounds @simd for i in 1:inp.n
    rnp.input_alloc[i] += inp.input[i]
  end
  return nothing
end

"""
    RateNoisyHomogeneousInput

Independent homogeneous Gaussian input for a rate population. For an isolated
linear Euler population, `μ` and `σ` are its exact stationary mean and standard
deviation. Rectification or saturation can change these observed statistics.
"""
struct RateNoisyHomogeneousInput <: RateInput
  n::Int64
  μ::Float64
  σ::Float64
  rand_alloc::Vector{Float64}
end

"""
    RateNoisyHomogeneousInput(n, μ, σ)

Create an independent noisy input for `n` target neurons.
"""
function RateNoisyHomogeneousInput(n::Int64,μ::Float64,σ::Float64)
  _check_noise_parameters(n,σ)
  return RateNoisyHomogeneousInput(n,μ,σ,fill(NaN,n))
end

"""
    noise_scale(inp::RateNoisyHomogeneousInput, dt, rnp::RateNeuralPopulation)

Return the standard deviation of the Gaussian drive added to the population
input at each numerical time step. This is the actual noise scale used by
`forward_signal!`; `inp.σ` is the resulting stationary rate standard deviation
only for an isolated, unclipped linear Euler population.
"""
function noise_scale(
    inp::RateNoisyHomogeneousInput,dt::Float64,rnp::RateNeuralPopulation)
  return _rate_noise_scale(rnp,dt,inp.σ)
end

"""
    forward_signal!(t_now, dt, rnp, inp::RateNoisyHomogeneousInput) -> nothing

Add one independent Gaussian input sample to `rnp.input_alloc`.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp::RateNeuralPopulation,inp::RateNoisyHomogeneousInput)
  _check_rate_input_dimension(rnp,inp.n)
  randn!(inp.rand_alloc)
  input_noise_scale = noise_scale(inp,dt,rnp)
  @inbounds @simd for i in 1:inp.n
    rnp.input_alloc[i] += inp.μ + input_noise_scale * inp.rand_alloc[i]
  end
  return nothing
end

"""
    RateSimpleCorrelatedInput

Homogeneous Gaussian input with one shared and one neuron-specific noise
component. For an isolated, unclipped linear Euler population, `μ` is the
stationary mean, `σ²` is every neuron's stationary variance, and `θ` is every
pair's Pearson correlation.
"""
struct RateSimpleCorrelatedInput <: RateInput
  n::Int64
  μ::Float64
  σ::Float64
  θ::Float64
  independent_alloc::Vector{Float64}
end

"""
    RateSimpleCorrelatedInput(n, μ, σ, θ)

Create correlated input with `θ=0` fully independent and `θ=1` fully shared.
"""
function RateSimpleCorrelatedInput(n::Int64,μ::Float64,σ::Float64,θ::Float64)
  _check_noise_parameters(n,σ)
  isfinite(θ) || throw(ArgumentError("θ must be finite; got $θ"))
  0.0 <= θ <= 1.0 || throw(ArgumentError("θ must lie in [0, 1]; got $θ"))
  return RateSimpleCorrelatedInput(n,μ,σ,θ,fill(NaN,n))
end

"""
    forward_signal!(t_now, dt, rnp, inp::RateSimpleCorrelatedInput) -> nothing

Add correlated Gaussian input using a shared component weighted by `sqrt(θ)`
and independent components weighted by `sqrt(1-θ)`.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp::RateNeuralPopulation,inp::RateSimpleCorrelatedInput)
  _check_rate_input_dimension(rnp,inp.n)
  randn!(inp.independent_alloc)
  shared_noise = randn()
  shared_scale = sqrt(inp.θ)
  independent_scale = sqrt(1.0 - inp.θ)
  noise_scale = _rate_noise_scale(rnp,dt,inp.σ)
  @inbounds @simd for i in 1:inp.n
    noise = shared_scale * shared_noise + independent_scale * inp.independent_alloc[i]
    rnp.input_alloc[i] += inp.μ + noise_scale * noise
  end
  return nothing
end

"""
    RateGeneralCorrelatedInput

Homogeneous Gaussian input with a user-specified Pearson correlation matrix.
For an isolated, unclipped linear Euler population, the stationary covariance
matrix is `σ² * correlation_levels`. The supplied diagonal is ignored and
treated as one.
"""
struct RateGeneralCorrelatedInput <: RateInput
  n::Int64
  μ::Float64
  σ::Float64
  correlation_levels::Matrix{Float64}
  correlation_factor::Matrix{Float64}
  rand_alloc::Vector{Float64}
  correlated_alloc::Vector{Float64}
end

"""
    RateGeneralCorrelatedInput(correlation_levels, μ, σ)

Create a correlated input from a symmetric positive-semidefinite matrix of
pairwise Pearson correlations. Off-diagonal entries must lie in `[0,1]`.
"""
function RateGeneralCorrelatedInput(
    correlation_levels::Matrix{Float64},μ::Float64,σ::Float64)
  n,n_columns = size(correlation_levels)
  n == n_columns || throw(DimensionMismatch(
    "correlation_levels must be square; got size $(size(correlation_levels))",
  ))
  _check_noise_parameters(n,σ)

  correlations = copy(correlation_levels)
  @inbounds for i in 1:n
    correlations[i,i] = 1.0
  end

  symmetry_tolerance = sqrt(eps(Float64))
  isapprox(correlations,transpose(correlations);
    atol=symmetry_tolerance,rtol=symmetry_tolerance) || throw(ArgumentError(
      "correlation_levels must be symmetric",
    ))

  @inbounds for j in 1:n
    for i in 1:j-1
      value = correlations[i,j]
      isfinite(value) || throw(ArgumentError(
        "Off-diagonal correlation levels must be finite",
      ))
      0.0 <= value <= 1.0 || throw(ArgumentError(
        "Off-diagonal correlation levels must lie in [0, 1]; got $value",
      ))
      symmetric_value = (correlations[i,j] + correlations[j,i]) / 2.0
      correlations[i,j] = symmetric_value
      correlations[j,i] = symmetric_value
    end
  end

  if n == 0
    correlation_factor = zeros(Float64,0,0)
  else
    decomposition = eigen(Symmetric(correlations))
    eigenvalue_scale = max(maximum(abs,decomposition.values),1.0)
    psd_tolerance = 100.0 * eps(Float64) * n * eigenvalue_scale
    minimum(decomposition.values) >= -psd_tolerance || throw(ArgumentError(
      "correlation_levels must be positive semidefinite",
    ))
    nonnegative_values = map(decomposition.values) do value
      value > psd_tolerance ? value : 0.0
    end
    correlation_factor = decomposition.vectors * Diagonal(sqrt.(nonnegative_values))
  end

  return RateGeneralCorrelatedInput(
    n,μ,σ,correlations,correlation_factor,fill(NaN,n),fill(NaN,n),
  )
end

"""
    forward_signal!(t_now, dt, rnp, inp::RateGeneralCorrelatedInput) -> nothing

Add one Gaussian sample having the requested correlation matrix.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp::RateNeuralPopulation,inp::RateGeneralCorrelatedInput)
  _check_rate_input_dimension(rnp,inp.n)
  randn!(inp.rand_alloc)
  mul!(inp.correlated_alloc,inp.correlation_factor,inp.rand_alloc)
  noise_scale = _rate_noise_scale(rnp,dt,inp.σ)
  @inbounds @simd for i in 1:inp.n
    rnp.input_alloc[i] += inp.μ + noise_scale * inp.correlated_alloc[i]
  end
  return nothing
end

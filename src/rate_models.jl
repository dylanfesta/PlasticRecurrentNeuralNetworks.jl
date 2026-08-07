#=
Component of PlasticRecurrentNeuralNetworks.jl package for the simulation of rate-based ReLu models

=#


"""
    RateNeuronType

Abstract parent for rate-neuron parameter types.

Concrete subtypes store intrinsic neuron parameters, such as the membrane/rate
time constant `τ`, and are carried by `RateNeuralPopulation`s.
"""
abstract type RateNeuronType <: NeuronType end  # RNT

"""
    RateNeuralPopulation

Abstract parent for rate populations with continuous-valued activity.

Concrete populations store the current rate vector and per-step work buffers.
"""
abstract type RateNeuralPopulation <: NeuralPopulation end  # RNP

"""
    RateSynapses

Abstract parent for synaptic operators between rate populations.

Connectivity follows the package convention `post <- pre`: weights and
statistics are indexed as `[post, pre]`.
"""
abstract type RateSynapses <: Synapses end  #  RSY

"""
    RateWeights

Abstract parent for rate-model weight containers.
"""
abstract type RateWeights <: Weights end  #  RW

"""
    RateRecorder

Abstract parent for recorders that sample rate-model state.
"""
abstract type RateRecorder <: Recorder end  #  RR


"""
    ExcitatoryRateNeuron(τ; rate_saturation=200.0)

Rate-neuron parameters for an excitatory population.

The field `τ` is the rate time constant used by `local_update!`, and
`rate_saturation` is the upper bound applied after each rate update. When this
population is the presynaptic side of a `RateLinearSynapses`, its synaptic drive
is added to the postsynaptic input buffer.
"""
struct ExcitatoryRateNeuron <: RateNeuronType
  τ::Float64
  rate_saturation::Float64
end

function ExcitatoryRateNeuron(τ::Float64; rate_saturation::Float64=200.0)
  return ExcitatoryRateNeuron(τ,rate_saturation)
end

"""
    InhibitoryRateNeuron(τ; rate_saturation=200.0)

Rate-neuron parameters for an inhibitory population.

The field `τ` is the rate time constant used by `local_update!`, and
`rate_saturation` is the upper bound applied after each rate update. When this
population is the presynaptic side of a `RateLinearSynapses`, its synaptic drive
is subtracted from the postsynaptic input buffer.
"""
struct InhibitoryRateNeuron <: RateNeuronType
  τ::Float64
  rate_saturation::Float64
end

function InhibitoryRateNeuron(τ::Float64; rate_saturation::Float64=200.0)
  return InhibitoryRateNeuron(τ,rate_saturation)
end

"""
    LinearRateNeuralPopulation

Population of rectified linear rate neurons.

Fields:
- `neuron_type`: rate-neuron parameter object, including the time constant `τ`.
- `n`: number of neurons.
- `rates_now`: current rates, length `n`.
- `input_alloc`: per-step input buffer accumulated by inputs and synapses.
- `utility_alloc`: scratch buffer reserved for algorithms that need temporary storage.

The rate update is Euler integration of `τ dr/dt = -r + input`, followed by
rectification at zero.
"""
struct LinearRateNeuralPopulation <: RateNeuralPopulation
  neuron_type::RateNeuronType
  n::Int64
  rates_now::Vector{Float64}
  input_alloc::Vector{Float64}
  utility_alloc::Vector{Float64}
end

"""
    LinearRateNeuralPopulation(neuron_type, n; initial_rates=nothing)

Create a `LinearRateNeuralPopulation` with zeroed input and utility buffers.

`initial_rates` may be:
- `nothing`: initialize all rates to `0.0`.
- a `Float64`: fill all rates with that value.
- a `Vector{Float64}` of length `n`: use that vector as `rates_now`.

The vector case keeps the supplied vector rather than copying it.
"""
function LinearRateNeuralPopulation(neuron_type::RateNeuronType,n::Int64;
    initial_rates::Union{Nothing,Float64,Vector{Float64}}=nothing)
  if isnothing(initial_rates)
    rates_now = fill(0.0,n)
  elseif isa(initial_rates,Float64)
    rates_now = fill(initial_rates,n)
  elseif isa(initial_rates,Vector{Float64})
    @assert length(initial_rates) == n "Length of initial_rates must be equal to n"
    rates_now = initial_rates
  else 
    error("Invalid type for initial_rates")
  end

  return LinearRateNeuralPopulation(
        neuron_type,
        n,
        rates_now,
        zeros(Float64,n),
        zeros(Float64,n)
    )
end

"""
    clean_up!(rnp::LinearRateNeuralPopulation) -> nothing

Reset the population input buffer to zero before the next simulation step.
"""
function clean_up!(rnp::LinearRateNeuralPopulation)
  rnp.input_alloc .= 0.0
  return nothing
end


"""
    local_update!(t_now, dt, rnp::LinearRateNeuralPopulation) -> nothing

Advance the population rates by one Euler step.

For each neuron, this applies
`r_new = r_old + (dt / τ) * (-r_old + input)` and then clamps values to
`[0, neuron_type.rate_saturation]`. `input_alloc` is read but not cleared; call
`clean_up!` at the beginning of each simulation step to clear accumulated input.
"""
function local_update!(t_now::Float64,dt::Float64,rnp::LinearRateNeuralPopulation)
  # linear rate neuron dynamics
  # dr/dt = -r + input(t)
  # r_new = r_old + dt * (-r_old + input(t))
  τ = rnp.neuron_type.τ
  rate_saturation = rnp.neuron_type.rate_saturation
  @inbounds @simd for i in 1:rnp.n
    _new_rate = rnp.rates_now[i] + (dt/τ) * (-rnp.rates_now[i] + rnp.input_alloc[i])
    if _new_rate > rate_saturation
      rnp.rates_now[i] = rate_saturation
    elseif _new_rate >= 0.0
      rnp.rates_now[i] = _new_rate
    else 
      rnp.rates_now[i] = 0.0
    end
  end
  return nothing
end


#############################
# ====== Synapses ========= #
#############################

"""
    RateLinearSynapses

Dense linear synapse between two rate populations.

`weights` has shape `(n_post, n_pre)` and follows `post <- pre`: `weights[i,j]`
is the magnitude of the connection from presynaptic neuron `j` to postsynaptic
neuron `i`. The sign of the contribution is determined by the presynaptic
population type: excitatory sources add drive, inhibitory sources subtract it.
"""
struct RateLinearSynapses <: RateSynapses
  n_pre::Int64
  n_post::Int64
  weights::Matrix{Float64}

  function RateLinearSynapses(weights::Matrix{Float64})
    n_post,n_pre = size(weights)
    return new(n_pre,n_post,weights)
  end
end

"""
    forward_signal!(t_now, dt, rnp_post, sy::RateLinearSynapses, rnp_pre) -> nothing

Accumulate synaptic drive from `rnp_pre` into `rnp_post.input_alloc`.

The synapse computes `weights * r_pre` with the package's `post <- pre`
orientation. Excitatory presynaptic populations add this drive; inhibitory
presynaptic populations subtract it.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp_post::RateNeuralPopulation,sy::RateLinearSynapses,rnp_pre::RateNeuralPopulation)
  if isa(rnp_pre.neuron_type,ExcitatoryRateNeuron)
    mul!(rnp_post.input_alloc,sy.weights,rnp_pre.rates_now,1.0,1.0)
  elseif isa(rnp_pre.neuron_type,InhibitoryRateNeuron)
    mul!(rnp_post.input_alloc,sy.weights,rnp_pre.rates_now,-1.0,1.0)
  end
  return nothing
end


############################
# ===== Recorders ======== #
############################

"""
    RCRate

Recorder for sampled rate trajectories from one rate population.

`times` stores recording times and `rates` stores one row per recorded sample.
Rows that have not been written are initialized to `NaN`; `krec` is the index of
the last written row.
"""
mutable struct RCRate <: RateRecorder
  population::RateNeuralPopulation
  dt::Float64
  t_end::Float64
  t_start::Float64
  times::Vector{Float64}
  rates::Matrix{Float64}
  krec::Int64
end


"""
    RCRate(population, t_end, dt; t_start=0.0)

Allocate a rate recorder for `population`.

The recorder samples at multiples of `dt` over `[t_start, t_end]`, inclusive up
to floating-point tolerance. Storage is allocated eagerly from the requested
time window.
"""
function RCRate(population::RateNeuralPopulation,t_end::Float64,dt::Float64;t_start=0.0)
  _nrec = ceil(Int,(t_end - t_start) / dt) + 1
  return RCRate(
    population,
    dt,
    t_end,
    t_start,
    fill(NaN,_nrec),
    fill(NaN,_nrec,population.n),
    0
  )
end


"""
    reset!(rec::RCRate) -> nothing

Clear all recorded samples and reset the write index.
"""
function reset!(rec::RCRate)
  rec.krec = 0
  fill!(rec.times,NaN)
  fill!(rec.rates,NaN)
  return nothing
end

"""
    record!(t_now, rec::RCRate) -> nothing

Record the population's current rates if `t_now` reaches the next recorder
sample time.

Calls outside `[t_start, t_end]` or repeated calls within the same recorder bin
are ignored. If storage is exhausted, a warning is emitted and no sample is
written.
"""
function record!(t_now::Float64,rec::RCRate)
  # if out of bonds, do nothing
  if (t_now<rec.t_start) || (t_now>rec.t_end)
    return nothing
  end
  _elapsed = t_now - rec.t_start
  _time_tol = 8 * eps(max(abs(_elapsed),abs(rec.dt),1.0))
  _krec_now = floor(Int,(_elapsed + _time_tol) / rec.dt) + 1
  # no discrete step in recorder dt, nothing to do
  if _krec_now <= rec.krec
    return nothing
  end
  # simulation longer than expected?
  if _krec_now > length(rec.times)
    @warn "Recorder full! Increase nrec to avoid data loss."
    return nothing
  end
  # store the rate, update the step
  rec.krec = _krec_now
  rec.times[rec.krec] = t_now
  copy!(view(rec.rates,rec.krec,:),rec.population.rates_now)
  return nothing
end

"""
    RCRateContent

Trimmed, read-oriented view of an `RCRate` recording.

`times` and `rates` include only written samples. `Tstart`, `Tend`, and `dt`
copy the recorder configuration.
"""
struct RCRateContent
  times::Vector{Float64}
  rates::Matrix{Float64}
  Tstart::Float64
  Tend::Float64
  dt::Float64
end

"""
    RCRateContent(rec::RCRate)

Return recorded rate data with unwritten trailing `NaN` rows removed.
"""
function RCRateContent(r::RCRate)
  _times = r.times
  _rates = r.rates
  idx_last = findfirst(isnan,_times)
  if isnothing(idx_last)
    return RCRateContent(_times,_rates,r.t_start,r.t_end,r.dt)
  else
    return RCRateContent(_times[1:idx_last-1],_rates[1:idx_last-1,:],r.t_start,r.t_end,r.dt)
  end
end

"""
    get_content(rec::RCRate)

Return an `RCRateContent` object for `rec`.
"""
function get_content(rec::RCRate)
  return RCRateContent(rec)
end



"""
    RCWeights

Recorder for synaptic weights between two populations in a rate model, just stores weight matrices.

`times` stores recording times 
`weights` is an array with dimensions `(n_rec,n_post, n_pre)` where `n_rec` is the number of recorded samples.

"""
mutable struct RCWeights <: RateRecorder
  weight_matrix_now::Matrix{Float64}
  dt::Float64
  t_end::Float64
  t_start::Float64
  times::Vector{Float64}
  weights::Array{Float64,3}
  krec::Int64
end

"""
    RCWeights(weight_matrix_now, t_end, dt; t_start=0.0)
    RCWeights(synapses, t_end, dt; t_start=0.0)

Allocate a weight recorder.

The matrix constructor stores `weight_matrix_now` directly, so later
modifications to the same matrix are recorded. The synapse constructor records
`synapses.weights`. Stored weights have dimensions `(n_rec, n_post, n_pre)`.
"""
function RCWeights(weight_matrix_now::Matrix{Float64},t_end::Float64,dt::Float64;t_start=0.0)
  _nrec = ceil(Int,(t_end - t_start) / dt) + 1
  return RCWeights(
    weight_matrix_now,
    dt,
    t_end,
    t_start,
    fill(NaN,_nrec),
    fill(NaN,_nrec,size(weight_matrix_now,1),size(weight_matrix_now,2)),
    0
  )
end

function RCWeights(synapses::RateSynapses,t_end::Float64,dt::Float64;t_start=0.0)
  return RCWeights(synapses.weights,t_end,dt;t_start=t_start)
end

function RCWeights(weight_matrix_now::Matrix{Float64},dt::Float64,t_end::Float64,t_start::Float64)
  return RCWeights(weight_matrix_now,t_end,dt;t_start=t_start)
end

"""
    reset!(rec::RCWeights) -> nothing

Clear all recorded weight samples and reset the write index.
"""
function reset!(rec::RCWeights)
  rec.krec = 0
  fill!(rec.times,NaN)
  fill!(rec.weights,NaN)
  return nothing
end

"""
    record!(t_now, rec::RCWeights) -> nothing

Record the current weight matrix if `t_now` reaches the next recorder sample
time.

Calls outside `[t_start, t_end]` or repeated calls within the same recorder bin
are ignored. If storage is exhausted, a warning is emitted and no sample is
written.
"""
function record!(t_now::Float64,rec::RCWeights)
  if (t_now<rec.t_start) || (t_now>rec.t_end)
    return nothing
  end
  _elapsed = t_now - rec.t_start
  _time_tol = 8 * eps(max(abs(_elapsed),abs(rec.dt),1.0))
  _krec_now = floor(Int,(_elapsed + _time_tol) / rec.dt) + 1
  if _krec_now <= rec.krec
    return nothing
  end
  if _krec_now > length(rec.times)
    @warn "Recorder full! Increase nrec to avoid data loss."
    return nothing
  end
  rec.krec = _krec_now
  rec.times[rec.krec] = t_now
  copy!(view(rec.weights,rec.krec,:,:),rec.weight_matrix_now)
  return nothing
end

"""
    RCWeightsContent

Trimmed, read-oriented view of an `RCWeights` recording.
"""
struct RCWeightsContent
  times::Vector{Float64}
  weights::Array{Float64,3}
  Tstart::Float64
  Tend::Float64
  dt::Float64
end

"""
    RCWeightsContent(rec::RCWeights)

Return recorded weight data with unwritten trailing rows removed.
"""
function RCWeightsContent(rec::RCWeights)
  return RCWeightsContent(
    rec.times[1:rec.krec],
    rec.weights[1:rec.krec,:,:],
    rec.t_start,
    rec.t_end,
    rec.dt
  )
end

"""
    get_content(rec::RCWeights)

Return an `RCWeightsContent` object for `rec`.
"""
function get_content(rec::RCWeights)
  return RCWeightsContent(rec)
end


###############################################
# ===== Mean and variance estimators ======== #
###############################################

# included in plasticity rules, updated by plasticity updates


"""
    RateMeanEstimator

Online exponentially weighted mean estimator for one rate population.

The estimator updates at interval `dt`, not necessarily every neural integration
step. It stores `propagation_factor = exp(-dt / τ)` and updates
`mean_now = a * mean_now + (1 - a) * rates_now` whenever the estimator interval
has elapsed.
"""
mutable struct RateMeanEstimator
  pop::RateNeuralPopulation
  n::Int64
  τ::Float64
  dt::Float64
  propagation_factor::Float64
  mean_now::Vector{Float64}
  t_last_update::Float64
end

"""
    RateMeanEstimator(pop, τ, dt)

Create a mean estimator for `pop` with memory time constant `τ` and update
interval `dt`.

The initial mean is zero and `t_last_update` is `-Inf`, so the first
`local_update!` call updates immediately.
"""
function RateMeanEstimator(pop::RateNeuralPopulation,τ::Float64,dt::Float64;
    initial_mean::Union{Vector{Float64},Float64}=0.0)
  if isa(initial_mean,Vector{Float64})
    @assert length(initial_mean) == pop.n "Length of initial_mean must be equal to pop.n"
    mean_now = initial_mean
  elseif isa(initial_mean,Float64)
    mean_now = fill(initial_mean,pop.n)
  else
    error("Invalid type for initial_mean")
  end 
  return RateMeanEstimator(
    pop,
    pop.n,
    τ,
    dt,
    exp(-dt / τ),
    mean_now,
    -Inf
  )
end


"""
    reset!(est::RateMeanEstimator) -> nothing

Reset the estimated mean to zero and allow the next `local_update!` call to
update immediately.
"""
function reset!(est::RateMeanEstimator)
  fill!(est.mean_now,0.0)
  est.t_last_update = -Inf
  return nothing
end

"""
    local_update!(t_now, dt, est::RateMeanEstimator) -> nothing

Update the mean trace if at least `est.dt` time has elapsed since the previous
estimator update.

The `dt` argument is accepted for the shared local-update interface; the
estimator's own interval `est.dt` determines the exponential propagation factor.
"""
function _update_rate_mean!(
    mean_now::Vector{Float64},
    rates_now::Vector{Float64},
    propagation_factor::Float64)
  input_factor = 1.0 - propagation_factor
  @inbounds @simd for i in eachindex(mean_now,rates_now)
    mean_now[i] = propagation_factor * mean_now[i] + input_factor * rates_now[i]
  end
  return nothing
end

function local_update!(t_now::Float64,dt::Float64,est::RateMeanEstimator)
  # update only if enough time has passed
  if t_now - est.t_last_update < est.dt
    return nothing
  end
  est.t_last_update = t_now
  _update_rate_mean!(est.mean_now,est.pop.rates_now,est.propagation_factor)
  return nothing
end


abstract type RateCovarianceAccumulator end

"""
    RateCovarianceEstimator

Online second-moment and covariance estimator for a pair of rate populations.
Matrices use the package convention `post <- pre`, so entry `[i,j]` corresponds
to post neuron `i` and pre neuron `j`.

The estimator stores the exponentially weighted second moment `Q` in
`second_moment_now` and the covariance `C` in `covariance_now`. Means are read
from externally supplied `RateMeanEstimator`s to keep first- and second-order
statistics modular.
"""
mutable struct RateCovarianceEstimator <: RateCovarianceAccumulator
  pop_post::RateNeuralPopulation
  pop_pre::RateNeuralPopulation
  n_post::Int64
  n_pre::Int64
  τ::Float64
  dt::Float64
  propagation_factor::Float64
  mean_post_estimator::RateMeanEstimator
  mean_pre_estimator::RateMeanEstimator
  second_moment_now::Matrix{Float64}
  covariance_now::Matrix{Float64}
  t_last_update::Float64
end

"""
    RateCovarianceEstimator(pop_post, pop_pre, mean_post_estimator, mean_pre_estimator, τ, dt)
    RateCovarianceEstimator(mean_post_estimator, mean_pre_estimator; τ=nothing, dt=nothing)

Construct a covariance estimator that tracks
`Q = E[r_post * r_pre']` and `C = Q - μ_post * μ_pre'`.

The initial second moment is seeded as `Q = μ_post * μ_pre'` from the
supplied mean estimators, so the initial covariance is zero without introducing
a negative transient at the first update.

The shorter constructor extracts the populations from the supplied mean
estimators. If `τ` or `dt` is omitted, that value is inferred from the mean
estimators and the post/pre mean estimators must agree.
"""
function RateCovarianceEstimator(
    pop_post::RateNeuralPopulation,
    pop_pre::RateNeuralPopulation,
    mean_post_estimator::RateMeanEstimator,
    mean_pre_estimator::RateMeanEstimator,
    τ::Float64,
    dt::Float64)
  @assert mean_post_estimator.pop === pop_post "Post mean estimator must track pop_post"
  @assert mean_pre_estimator.pop === pop_pre "Pre mean estimator must track pop_pre"
  est = RateCovarianceEstimator(
    pop_post,
    pop_pre,
    pop_post.n,
    pop_pre.n,
    τ,
    dt,
    exp(-dt / τ),
    mean_post_estimator,
    mean_pre_estimator,
    fill(0.0,pop_post.n,pop_pre.n),
    fill(0.0,pop_post.n,pop_pre.n),
    -Inf
  )
  reset!(est)
  return est
end

function RateCovarianceEstimator(
    mean_post_estimator::RateMeanEstimator,
    mean_pre_estimator::RateMeanEstimator;
    τ::Union{Nothing,Float64}=nothing,
    dt::Union{Nothing,Float64}=nothing)
  if isnothing(τ)
    @assert mean_post_estimator.τ == mean_pre_estimator.τ "Cannot infer covariance τ: post and pre mean estimators use different τ values"
    τ = mean_post_estimator.τ
  end
  if isnothing(dt)
    @assert mean_post_estimator.dt == mean_pre_estimator.dt "Cannot infer covariance dt: post and pre mean estimators use different dt values"
    dt = mean_post_estimator.dt
  end

  return RateCovarianceEstimator(
    mean_post_estimator.pop,
    mean_pre_estimator.pop,
    mean_post_estimator,
    mean_pre_estimator,
    τ,
    dt,
  )
end


"""
    reset!(est::RateCovarianceEstimator) -> nothing

Reset the covariance to zero and seed the second moment from the current mean
estimates as `Q = μ_post * μ_pre'`.

The referenced mean estimators are not reset.
"""
function reset!(est::RateCovarianceEstimator)
  post_means = est.mean_post_estimator.mean_now
  pre_means = est.mean_pre_estimator.mean_now
  @inbounds for j in 1:est.n_pre
    pre_mean = pre_means[j]
    for i in 1:est.n_post
      est.second_moment_now[i,j] = post_means[i] * pre_mean
      est.covariance_now[i,j] = 0.0
    end
  end
  est.t_last_update = -Inf
  return nothing
end


"""
    local_update!(t_now, dt, est::RateCovarianceEstimator) -> nothing

Update `second_moment_now` and `covariance_now` if at least `est.dt` time has
elapsed since the previous covariance update.

The second moment uses
`Q = b * Q + (1 - b) * (r_post * r_pre')`, with
`b = est.propagation_factor`. The covariance is then computed from the current
mean-estimator states as `C = Q - μ_post * μ_pre'`.
"""
function _update_rate_covariance!(
    second_moment_now::Matrix{Float64},
    covariance_now::Matrix{Float64},
    post_rates::Vector{Float64},
    pre_rates::Vector{Float64},
    post_means::Vector{Float64},
    pre_means::Vector{Float64},
    propagation_factor::Float64)
  input_factor = 1.0 - propagation_factor
  n_post,n_pre = size(second_moment_now)
  @inbounds for j in 1:n_pre
    pre_rate = pre_rates[j]
    pre_mean = pre_means[j]
    for i in 1:n_post
      second_moment = propagation_factor * second_moment_now[i,j] +
        input_factor * post_rates[i] * pre_rate
      second_moment_now[i,j] = second_moment
      covariance_now[i,j] = second_moment - post_means[i] * pre_mean
    end
  end
  return nothing
end

function local_update!(t_now::Float64,dt::Float64,est::RateCovarianceEstimator)
  # update only if enough time has passed
  if t_now - est.t_last_update < est.dt
    return nothing
  end
  est.t_last_update = t_now
  _update_rate_covariance!(
    est.second_moment_now,
    est.covariance_now,
    est.pop_post.rates_now,
    est.pop_pre.rates_now,
    est.mean_post_estimator.mean_now,
    est.mean_pre_estimator.mean_now,
    est.propagation_factor,
  )
  return nothing
end

"""
    CovarianceTransposed(covariance_estimator)

View a covariance estimator with post/pre orientation transposed.

This is useful when two plasticity rules should use the same running covariance
estimate in opposite synaptic directions. Access to covariance entries should go
through `covariance_at`, which returns `C[i,j]` for a plain estimator and
`C[j,i]` for a `CovarianceTransposed` wrapper.
"""
struct CovarianceTransposed <: RateCovarianceAccumulator
  covariance_estimator::RateCovarianceEstimator
end

covariance_pop_post(est::RateCovarianceEstimator) = est.pop_post
covariance_pop_pre(est::RateCovarianceEstimator) = est.pop_pre
covariance_n_post(est::RateCovarianceEstimator) = est.n_post
covariance_n_pre(est::RateCovarianceEstimator) = est.n_pre
covariance_at(est::RateCovarianceEstimator,i::Int64,j::Int64) =
  est.covariance_now[i,j]

covariance_pop_post(est::CovarianceTransposed) =
  est.covariance_estimator.pop_pre
covariance_pop_pre(est::CovarianceTransposed) =
  est.covariance_estimator.pop_post
covariance_n_post(est::CovarianceTransposed) =
  est.covariance_estimator.n_pre
covariance_n_pre(est::CovarianceTransposed) =
  est.covariance_estimator.n_post
covariance_at(est::CovarianceTransposed,i::Int64,j::Int64) =
  est.covariance_estimator.covariance_now[j,i]

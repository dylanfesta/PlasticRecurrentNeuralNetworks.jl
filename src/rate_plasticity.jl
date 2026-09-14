###################################
# ===== Plasticity rules ======== #
###################################

"""
    RatePlasticity

Abstract parent for plasticity rules acting on rate-model state.

Every concrete rate-plasticity rule has an `is_active` reference. Use
[`plasticity_on!`](@ref) and [`plasticity_off!`](@ref) to change that reference
without replacing the immutable rule object.
"""
abstract type RatePlasticity <: Plasticity end

"""
    plasticity_on!(rule::RatePlasticity) -> nothing

Enable weight updates for `rule`. If multiple rules share the same
`is_active` reference, enabling any one of them enables all of them.
"""
function plasticity_on!(rule::RatePlasticity)
  rule.is_active[] = true
  return nothing
end

"""
    plasticity_off!(rule::RatePlasticity) -> nothing

Disable weight updates for `rule`. Its update schedule is frozen until the
rule is enabled again. Estimators registered with `RecurrentNetwork` continue
to update independently.
"""
function plasticity_off!(rule::RatePlasticity)
  rule.is_active[] = false
  return nothing
end

"""
    RatePlasticityHomeostaticScaling
    RatePlasticityHomeostaticScaling(pop_post, synapses_post_pre, pop_pre, α, s, Δt, learning_rate, rate_estimator_post; is_active=Ref(true), w_min=1E-8, w_max=Inf)

Homeostatic scaling plasticity rule for a pair of rate populations.
The update is `w <- w + Δt * learning_rate * w * r_post * s * (mean_post - α)`
every `Δt` seconds. `learning_rate` is interpreted per unit time.

The compulsory sign `s` must be `-1` for excitatory presynaptic populations
and `+1` for inhibitory presynaptic populations, matching the Hawkes rule.
Only these two values are accepted; the caller selects the sign explicitly.
Weights represent positive connection strengths. Above the postsynaptic target
`α`, excitatory strengths decrease and inhibitory strengths increase; below
target, these directions reverse (for positive rates and learning rate).
All incoming weights to a given postsynaptic neuron within this rule receive
the same multiplicative factor before clipping.

Pass a shared `is_active` reference to coordinate this rule with other rules.
Estimators update independently through `RecurrentNetwork.estimators` while the
rule is inactive.

Exactly zero weights are skipped. Initialize connections that should be plastic
with positive strengths and use a small positive `w_min` to keep them plastic.
"""
struct RatePlasticityHomeostaticScaling <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  α::Float64
  s::Float64
  Δt::Float64
  learning_rate::Float64
  rate_estimator_post::RateMeanEstimator
  t_last_update::Base.RefValue{Float64}
  is_active::Base.RefValue{Bool}
  w_min::Float64
  w_max::Float64
end

function RatePlasticityHomeostaticScaling(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    α::Float64,
    s::Real,
    Δt::Float64,
    learning_rate::Float64,
    rate_estimator_post::RateMeanEstimator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true)
)
  if !(s == 1 || s == -1)
    throw(ArgumentError("s must be +1 (inhibitory pre) or -1 (excitatory pre)"))
  end
  @assert rate_estimator_post.pop === pop_post "Rate estimator must track the postsynaptic population"
  return RatePlasticityHomeostaticScaling(
    pop_pre,
    pop_post,
    synapses_post_pre,
    α,
    Float64(s),
    Δt,
    learning_rate,
    rate_estimator_post,
    Ref(-Inf),
    is_active,
    w_min,
    w_max
  )
end


function _update_homeostatic_scaling!(
    weights::Matrix{Float64},
    post_rates::Vector{Float64},
    post_means::Vector{Float64},
    α::Float64,
    s::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * w_old * post_rates[i] * s * (post_means[i] - α)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityHomeostaticScaling)
  rule.is_active[] || return nothing
  # update only if enough time has passed
  if t_now - rule.t_last_update[] < rule.Δt
    return nothing
  end
  rule.t_last_update[] = t_now
  _update_homeostatic_scaling!(
    rule.synapses_post_pre.weights,
    rule.pop_post.rates_now,
    rule.rate_estimator_post.mean_now,
    rule.α,
    rule.s,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

"""
    RatePlasticityCovariance
    RatePlasticityCovariance(pop_post, synapses_post_pre, pop_pre, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)
    RatePlasticityCovariance(pop_post, synapses_post_pre, pop_pre, B, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)

Second-order covariance plasticity rule for a pair of rate populations:
`w <- w + Δt * learning_rate * (C_post_pre(t) + B * μ_post(t) * μ_pre(t) - α_leak * w)`.
The rule updates every `Δt` seconds, and `learning_rate` is interpreted per
unit time, so each update is internally scaled by `Δt`.

`C` is the covariance between the pre- and postsynaptic units, and `μ` is the
running mean rate for those same units. The mean estimators are read from the
covariance estimator.

The supplied covariance and mean estimators are external to the rule and should
be listed once in `RecurrentNetwork.estimators`. An inactive rule does not
change weights or advance its plasticity schedule.

The weight-leak coefficient `α_leak` defaults to zero, preserving the original rule.

By convention, the rule acts only on nonzero weights.
"""
struct RatePlasticityCovariance <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  B::Float64
  α_leak::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  t_last_update::Base.RefValue{Float64}
  is_active::Base.RefValue{Bool}
  w_min::Float64
  w_max::Float64
end

function RatePlasticityCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true)
)
  return RatePlasticityCovariance(
    pop_post,
    synapses_post_pre,
    pop_pre,
    0.0,
    Δt,
    learning_rate,
    covariance_estimator;
    α_leak=α_leak,
    w_min=w_min,
    w_max=w_max,
    is_active=is_active,
  )
end

function RatePlasticityCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    B::Float64,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true)
)
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  return RatePlasticityCovariance(
    pop_pre,
    pop_post,
    synapses_post_pre,
    B,
    α_leak,
    Δt,
    learning_rate,
    covariance_estimator,
    Ref(-Inf),
    is_active,
    w_min,
    w_max
  )
end

function _update_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    rates_pre::Vector{Float64},
    rates_post::Vector{Float64},
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  n_post,n_pre = size(weights)
  if B == 0.0
    @inbounds for j in 1:n_pre
      for i in 1:n_post
        w_old = weights[i,j]
        if w_old == 0.0
          continue
        end
        w_new = w_old + effective_learning_rate * (covariance_now[i,j] - α_leak * w_old)
        weights[i,j] = clamp(w_new,w_min,w_max)
      end
    end
    return nothing
  end
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * (covariance_now[i,j] + B * rates_post[i] * rates_pre[j] - α_leak * w_old)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function _update_covariance_plasticity_transposed!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    rates_pre::Vector{Float64},
    rates_post::Vector{Float64},
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  n_post,n_pre = size(weights)
  if B == 0.0
    @inbounds for j in 1:n_pre
      for i in 1:n_post
        w_old = weights[i,j]
        if w_old == 0.0
          continue
        end
        w_new = w_old + effective_learning_rate * (covariance_now[j,i] - α_leak * w_old)
        weights[i,j] = clamp(w_new,w_min,w_max)
      end
    end
    return nothing
  end
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * (covariance_now[j,i] + B * rates_post[i] * rates_pre[j] - α_leak * w_old)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function _update_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::RateCovarianceEstimator,
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_covariance_plasticity!(
    weights,
    covariance_estimator.covariance_now,
    covariance_estimator.mean_pre_estimator.mean_now,
    covariance_estimator.mean_post_estimator.mean_now,
    B,
    α_leak,
    effective_learning_rate,
    w_min,
    w_max,
  )
end

function _update_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::CovarianceTransposed,
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_covariance_plasticity_transposed!(
    weights,
    covariance_estimator.covariance_estimator.covariance_now,
    covariance_estimator.covariance_estimator.mean_post_estimator.mean_now,
    covariance_estimator.covariance_estimator.mean_pre_estimator.mean_now,
    B,
    α_leak,
    effective_learning_rate,
    w_min,
    w_max,
  )
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityCovariance)
  rule.is_active[] || return nothing
  # update only if enough time has passed
  if t_now - rule.t_last_update[] < rule.Δt
    return nothing
  end
  rule.t_last_update[] = t_now
  _update_covariance_plasticity!(
    rule.synapses_post_pre.weights,
    rule.covariance_estimator,
    rule.B,
    rule.α_leak,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

"""
    RatePlasticityScaledCovariance
    RatePlasticityScaledCovariance(pop_post, synapses_post_pre, pop_pre, scale_matrix, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)
    RatePlasticityScaledCovariance(pop_post, synapses_post_pre, pop_pre, scale_matrix, B, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)

Covariance-based plasticity rule with an arbitrary pre-post scaling factor:
`w <- w + Δt * learning_rate * A_post_pre * (C_post_pre(t) + B * μ_post(t) * μ_pre(t) - α_leak * w)`.
The rule updates every `Δt` seconds, and `learning_rate` is interpreted per
unit time, so each update is internally scaled by `Δt`. The running means are
read from the covariance estimator.

The `A_post_pre` scale matrix can act as a mask or, for example, as a
distance-based scaling factor. See [`generate_ring_topology`](@ref).

The covariance estimator is updated externally through
`RecurrentNetwork.estimators`, including while this rule is inactive.

The weight-leak coefficient `α_leak` defaults to zero, preserving the original rule.

By convention, the rule acts only on nonzero weights. Entries whose scale is
zero are also skipped, including the leak. Scaling applies to the entire update.
"""
struct RatePlasticityScaledCovariance <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  B::Float64
  α_leak::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  scale_matrix::Matrix{Float64}
  t_last_update::Base.RefValue{Float64}
  is_active::Base.RefValue{Bool}
  w_min::Float64
  w_max::Float64
end


function RatePlasticityScaledCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    scale_matrix::Matrix{Float64},
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true)
)
  return RatePlasticityScaledCovariance(
    pop_post,
    synapses_post_pre,
    pop_pre,
    scale_matrix,
    0.0,
    Δt,
    learning_rate,
    covariance_estimator;
    α_leak=α_leak,
    w_min=w_min,
    w_max=w_max,
    is_active=is_active,
  )
end

function RatePlasticityScaledCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    scale_matrix::Matrix{Float64},
    B::Float64,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true)
)
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  @assert size(scale_matrix) == (pop_post.n,pop_pre.n) "Scale matrix dimensions must match (pop_post.n, pop_pre.n)"
  return RatePlasticityScaledCovariance(
    pop_pre,
    pop_post,
    synapses_post_pre,
    B,
    α_leak,
    Δt,
    learning_rate,
    covariance_estimator,
    scale_matrix,
    Ref(-Inf),
    is_active,
    w_min,
    w_max
  )
end

function _update_scaled_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    scale_matrix::Matrix{Float64},
    rates_pre::Vector{Float64},
    rates_post::Vector{Float64},
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  n_post,n_pre = size(weights)
  if B == 0.0
    @inbounds for j in 1:n_pre
      for i in 1:n_post
        w_old = weights[i,j]
        scale = scale_matrix[i,j]
        if (w_old == 0.0) || (scale == 0.0)
          continue
        end
        w_new = w_old + effective_learning_rate * scale * (covariance_now[i,j] - α_leak * w_old)
        weights[i,j] = clamp(w_new,w_min,w_max)
      end
    end
    return nothing
  end
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      scale = scale_matrix[i,j]
      if (w_old == 0.0) || (scale == 0.0)
        continue
      end
      w_new = w_old + effective_learning_rate * scale * (covariance_now[i,j] + B * rates_post[i] * rates_pre[j] - α_leak * w_old)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function _update_scaled_covariance_plasticity_transposed!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    scale_matrix::Matrix{Float64},
    rates_pre::Vector{Float64},
    rates_post::Vector{Float64},
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  n_post,n_pre = size(weights)
  if B == 0.0
    @inbounds for j in 1:n_pre
      for i in 1:n_post
        w_old = weights[i,j]
        scale = scale_matrix[i,j]
        if (w_old == 0.0) || (scale == 0.0)
          continue
        end
        w_new = w_old + effective_learning_rate * scale * (covariance_now[j,i] - α_leak * w_old)
        weights[i,j] = clamp(w_new,w_min,w_max)
      end
    end
    return nothing
  end
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      scale = scale_matrix[i,j]
      if (w_old == 0.0) || (scale == 0.0)
        continue
      end
      w_new = w_old + effective_learning_rate * scale * (covariance_now[j,i] + B * rates_post[i] * rates_pre[j] - α_leak * w_old)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function _update_scaled_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::RateCovarianceEstimator,
    scale_matrix::Matrix{Float64},
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_scaled_covariance_plasticity!(
    weights,
    covariance_estimator.covariance_now,
    scale_matrix,
    covariance_estimator.mean_pre_estimator.mean_now,
    covariance_estimator.mean_post_estimator.mean_now,
    B,
    α_leak,
    effective_learning_rate,
    w_min,
    w_max,
  )
end

function _update_scaled_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::CovarianceTransposed,
    scale_matrix::Matrix{Float64},
    B::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_scaled_covariance_plasticity_transposed!(
    weights,
    covariance_estimator.covariance_estimator.covariance_now,
    scale_matrix,
    covariance_estimator.covariance_estimator.mean_post_estimator.mean_now,
    covariance_estimator.covariance_estimator.mean_pre_estimator.mean_now,
    B,
    α_leak,
    effective_learning_rate,
    w_min,
    w_max,
  )
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityScaledCovariance)
  rule.is_active[] || return nothing
  # update only if enough time has passed
  if t_now - rule.t_last_update[] < rule.Δt
    return nothing
  end
  rule.t_last_update[] = t_now
  _update_scaled_covariance_plasticity!(
    rule.synapses_post_pre.weights,
    rule.covariance_estimator,
    rule.scale_matrix,
    rule.B,
    rule.α_leak,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

"""
    RatePlasticitySQRC
    RatePlasticitySQRC(pop_post, synapses_post_pre, pop_pre, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)
    RatePlasticitySQRC(pop_post, synapses_post_pre, pop_pre, B, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)

Signed-square-root covariance plasticity. This is equivalent to
[`RatePlasticityCovariance`](@ref), except that each covariance `C` is replaced
by `sign(C) * sqrt(abs(C))`. The mean-product and leak terms are unchanged.
"""
struct RatePlasticitySQRC <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  B::Float64
  α_leak::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  t_last_update::Base.RefValue{Float64}
  is_active::Base.RefValue{Bool}
  w_min::Float64
  w_max::Float64
end

function RatePlasticitySQRC(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true))
  return RatePlasticitySQRC(
    pop_post,synapses_post_pre,pop_pre,0.0,Δt,learning_rate,covariance_estimator;
    α_leak=α_leak,w_min=w_min,w_max=w_max,is_active=is_active,
  )
end

function RatePlasticitySQRC(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    B::Float64,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true))
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  return RatePlasticitySQRC(
    pop_pre,pop_post,synapses_post_pre,B,α_leak,Δt,learning_rate,
    covariance_estimator,Ref(-Inf),is_active,w_min,w_max,
  )
end

"""
    RatePlasticityScaledSQRC
    RatePlasticityScaledSQRC(pop_post, synapses_post_pre, pop_pre, scale_matrix, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)
    RatePlasticityScaledSQRC(pop_post, synapses_post_pre, pop_pre, scale_matrix, B, Δt, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)

Scaled signed-square-root covariance plasticity. This is equivalent to
[`RatePlasticityScaledCovariance`](@ref), with each covariance `C` replaced by
`sign(C) * sqrt(abs(C))`.
"""
struct RatePlasticityScaledSQRC <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  B::Float64
  α_leak::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  scale_matrix::Matrix{Float64}
  t_last_update::Base.RefValue{Float64}
  is_active::Base.RefValue{Bool}
  w_min::Float64
  w_max::Float64
end

function RatePlasticityScaledSQRC(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    scale_matrix::Matrix{Float64},
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true))
  return RatePlasticityScaledSQRC(
    pop_post,synapses_post_pre,pop_pre,scale_matrix,0.0,Δt,learning_rate,
    covariance_estimator;α_leak=α_leak,w_min=w_min,w_max=w_max,is_active=is_active,
  )
end

function RatePlasticityScaledSQRC(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    scale_matrix::Matrix{Float64},
    B::Float64,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true))
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  @assert size(scale_matrix) == (pop_post.n,pop_pre.n) "Scale matrix dimensions must match (pop_post.n, pop_pre.n)"
  return RatePlasticityScaledSQRC(
    pop_pre,pop_post,synapses_post_pre,B,α_leak,Δt,learning_rate,
    covariance_estimator,scale_matrix,Ref(-Inf),is_active,w_min,w_max,
  )
end

@inline _signed_sqrt(covariance::Float64) = copysign(sqrt(abs(covariance)),covariance)

function _update_sqrc_plasticity!(
    weights::Matrix{Float64}, covariance_now::Matrix{Float64},
    rates_pre::Vector{Float64}, rates_post::Vector{Float64}, B::Float64,
    α_leak::Float64, effective_learning_rate::Float64,
    w_min::Float64, w_max::Float64; transposed::Bool=false,
    scale_matrix::Union{Nothing,Matrix{Float64}}=nothing)
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      scale = isnothing(scale_matrix) ? 1.0 : scale_matrix[i,j]
      if (w_old == 0.0) || (scale == 0.0)
        continue
      end
      covariance = transposed ? covariance_now[j,i] : covariance_now[i,j]
      update = _signed_sqrt(covariance) - α_leak * w_old
      if B != 0.0
        update += B * rates_post[i] * rates_pre[j]
      end
      weights[i,j] = clamp(w_old + effective_learning_rate * scale * update,w_min,w_max)
    end
  end
  return nothing
end

function _update_sqrc_plasticity!(weights::Matrix{Float64}, estimator::RateCovarianceEstimator,
    B::Float64, α_leak::Float64, learning_rate::Float64, w_min::Float64, w_max::Float64;
    scale_matrix::Union{Nothing,Matrix{Float64}}=nothing)
  return _update_sqrc_plasticity!(
    weights,estimator.covariance_now,estimator.mean_pre_estimator.mean_now,
    estimator.mean_post_estimator.mean_now,B,α_leak,learning_rate,w_min,w_max;
    scale_matrix=scale_matrix,
  )
end

function _update_sqrc_plasticity!(weights::Matrix{Float64}, estimator::CovarianceTransposed,
    B::Float64, α_leak::Float64, learning_rate::Float64, w_min::Float64, w_max::Float64;
    scale_matrix::Union{Nothing,Matrix{Float64}}=nothing)
  covariance_estimator = estimator.covariance_estimator
  return _update_sqrc_plasticity!(
    weights,covariance_estimator.covariance_now,
    covariance_estimator.mean_post_estimator.mean_now,
    covariance_estimator.mean_pre_estimator.mean_now,B,α_leak,learning_rate,w_min,w_max;
    transposed=true,scale_matrix=scale_matrix,
  )
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticitySQRC)
  rule.is_active[] || return nothing
  if t_now - rule.t_last_update[] < rule.Δt
    return nothing
  end
  rule.t_last_update[] = t_now
  _update_sqrc_plasticity!(
    rule.synapses_post_pre.weights,rule.covariance_estimator,rule.B,rule.α_leak,
    rule.learning_rate * rule.Δt,rule.w_min,rule.w_max,
  )
  return nothing
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityScaledSQRC)
  rule.is_active[] || return nothing
  if t_now - rule.t_last_update[] < rule.Δt
    return nothing
  end
  rule.t_last_update[] = t_now
  _update_sqrc_plasticity!(
    rule.synapses_post_pre.weights,rule.covariance_estimator,rule.B,rule.α_leak,
    rule.learning_rate * rule.Δt,rule.w_min,rule.w_max;
    scale_matrix=rule.scale_matrix,
  )
  return nothing
end


"""
    RatePlasticityCovarianceQuadraticallyStabilized
    RatePlasticityCovarianceQuadraticallyStabilized(pop_post, synapses_post_pre, pop_pre, Δt, α1, α2, learning_rate, covariance_estimator; α_leak=0.0, is_active=Ref(true), w_min=1E-8, w_max=Inf)

Covariance-based plasticity rule with an additional stabilizing term that
scales quadratically with the weight.
`w <- w + Δt * learning_rate * (α1 * C_post_pre(t) + α2 * w^2 - α_leak * w)`.
The rule updates every Δt seconds. `learning_rate` is interpreted per unit time,
so each update is internally scaled by Δt.
The weight-leak coefficient `α_leak` defaults to zero, preserving the original rule.

α1 and α2 are intended to be of opposite sign, and depend on the effect on the neuron on covariance.

For example, an inhibitory neuron will have α1 < 0 and α2 > 0, so that the covariance term will tend to decrease the weight,
while the quadratic term will tend to increase it, stabilizing the weight at a finite value.

The covariance estimator is updated externally through
`RecurrentNetwork.estimators`, including while this rule is inactive.

IMPORTANT: by convention the rule acts only on weights > 0 . So you must initialize all weights that you want plastic to a small
positive value, also making sure that w_min > 0 (and very small).
"""
struct RatePlasticityCovarianceQuadraticallyStabilized <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  α1::Float64
  α2::Float64
  α_leak::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  t_last_update::Base.RefValue{Float64}
  is_active::Base.RefValue{Bool}
  w_min::Float64
  w_max::Float64
end


function RatePlasticityCovarianceQuadraticallyStabilized(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    Δt::Float64,
    α1::Float64,
    α2::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    α_leak::Float64=0.0,
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true)
)
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  return RatePlasticityCovarianceQuadraticallyStabilized(
    pop_pre,
    pop_post,
    synapses_post_pre,
    α1,
    α2,
    α_leak,
    Δt,
    learning_rate,
    covariance_estimator,
    Ref(-Inf),
    is_active,
    w_min,
    w_max
  )
end


function _update_quadratic_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    α1::Float64,
    α2::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * (α1 * covariance_now[i,j] + α2 * w_old^2 - α_leak * w_old)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function _update_quadratic_covariance_plasticity_transposed!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    α1::Float64,
    α2::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * (α1 * covariance_now[j,i] + α2 * w_old^2 - α_leak * w_old)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function _update_quadratic_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::RateCovarianceEstimator,
    α1::Float64,
    α2::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_quadratic_covariance_plasticity!(
    weights,
    covariance_estimator.covariance_now,
    α1,
    α2,
    α_leak,
    effective_learning_rate,
    w_min,
    w_max,
  )
end

function _update_quadratic_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::CovarianceTransposed,
    α1::Float64,
    α2::Float64,
    α_leak::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_quadratic_covariance_plasticity_transposed!(
    weights,
    covariance_estimator.covariance_estimator.covariance_now,
    α1,
    α2,
    α_leak,
    effective_learning_rate,
    w_min,
    w_max,
  )
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityCovarianceQuadraticallyStabilized)
  rule.is_active[] || return nothing
  # update only if enough time has passed
  if t_now - rule.t_last_update[] < rule.Δt
    return nothing
  end
  rule.t_last_update[] = t_now
  _update_quadratic_covariance_plasticity!(
    rule.synapses_post_pre.weights,
    rule.covariance_estimator,
    rule.α1,
    rule.α2,
    rule.α_leak,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

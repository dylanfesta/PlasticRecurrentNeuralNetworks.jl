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
    RatePlasticityHomeostaticScaling(pop_post, synapses_post_pre, pop_pre, α, Δt, learning_rate, rate_estimator_post; is_active=Ref(true), w_min=1E-8, w_max=Inf)

Homeostatic scaling plasticity rule for a pair of rate populations.
the rule is w <- w + Δt * w * learning rate * r_post(t) *( α - r_post_mean_estimator(t) )
With an update every Δt seconds. `learning_rate` is interpreted per unit time,
so each update is internally scaled by the elapsed plasticity interval.

Note that this rule depends only on the postsynaptic rate, therefore it scales all incoming weights of the same amount.

Pass a shared `is_active` reference to coordinate this rule with other rules.
Estimators update independently through `RecurrentNetwork.estimators` while the
rule is inactive.

IMPORTANT: by convention the rule acts only on weights > 0 . So you must initialize all weights that you want plastic to a small
positive value, also making sure that w_min > 0 (and very small).
"""
struct RatePlasticityHomeostaticScaling <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  α::Float64
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
    Δt::Float64,
    learning_rate::Float64,
    rate_estimator_post::RateMeanEstimator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf,
    is_active::Base.RefValue{Bool}=Ref(true)
)
  @assert rate_estimator_post.pop === pop_post "Rate estimator must track the postsynaptic population"
  return RatePlasticityHomeostaticScaling(
    pop_pre,
    pop_post,
    synapses_post_pre,
    α,
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
      w_new = w_old + effective_learning_rate * w_old * post_rates[i] * (α - post_means[i])
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
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

"""
    RatePlasticityCovariance
    RatePlasticityCovariance(pop_post, synapses_post_pre, pop_pre, Δt, learning_rate, covariance_estimator; is_active=Ref(true), w_min=1E-8, w_max=Inf)
    RatePlasticityCovariance(pop_post, synapses_post_pre, pop_pre, B, Δt, learning_rate, covariance_estimator; is_active=Ref(true), w_min=1E-8, w_max=Inf)

Second-order covariance plasticity rule for a pair of rate populations:
`w <- w + Δt * learning_rate * (C_post_pre(t) + B * μ_post(t) * μ_pre(t))`.
The rule updates every `Δt` seconds, and `learning_rate` is interpreted per
unit time, so each update is internally scaled by `Δt`.

`C` is the covariance between the pre- and postsynaptic units, and `μ` is the
running mean rate for those same units. The mean estimators are read from the
covariance estimator.

The supplied covariance and mean estimators are external to the rule and should
be listed once in `RecurrentNetwork.estimators`. An inactive rule does not
change weights or advance its plasticity schedule.

By convention, the rule acts only on nonzero weights.
"""
struct RatePlasticityCovariance <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  B::Float64
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
        w_new = w_old + effective_learning_rate * covariance_now[i,j]
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
      w_new = w_old + effective_learning_rate * (covariance_now[i,j] + B * rates_post[i] * rates_pre[j])
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
        w_new = w_old + effective_learning_rate * covariance_now[j,i]
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
      w_new = w_old + effective_learning_rate * (covariance_now[j,i] + B * rates_post[i] * rates_pre[j])
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function _update_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::RateCovarianceEstimator,
    B::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_covariance_plasticity!(
    weights,
    covariance_estimator.covariance_now,
    covariance_estimator.mean_pre_estimator.mean_now,
    covariance_estimator.mean_post_estimator.mean_now,
    B,
    effective_learning_rate,
    w_min,
    w_max,
  )
end

function _update_covariance_plasticity!(
    weights::Matrix{Float64},
    covariance_estimator::CovarianceTransposed,
    B::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_covariance_plasticity_transposed!(
    weights,
    covariance_estimator.covariance_estimator.covariance_now,
    covariance_estimator.covariance_estimator.mean_post_estimator.mean_now,
    covariance_estimator.covariance_estimator.mean_pre_estimator.mean_now,
    B,
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
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

"""
    RatePlasticityScaledCovariance
    RatePlasticityScaledCovariance(pop_post, synapses_post_pre, pop_pre, scale_matrix, Δt, learning_rate, covariance_estimator; is_active=Ref(true), w_min=1E-8, w_max=Inf)
    RatePlasticityScaledCovariance(pop_post, synapses_post_pre, pop_pre, scale_matrix, B, Δt, learning_rate, covariance_estimator; is_active=Ref(true), w_min=1E-8, w_max=Inf)

Covariance-based plasticity rule with an arbitrary pre-post scaling factor:
`w <- w + Δt * learning_rate * A_post_pre * (C_post_pre(t) + B * μ_post(t) * μ_pre(t))`.
The rule updates every `Δt` seconds, and `learning_rate` is interpreted per
unit time, so each update is internally scaled by `Δt`. The running means are
read from the covariance estimator.

The `A_post_pre` scale matrix can act as a mask or, for example, as a
distance-based scaling factor. See [`generate_ring_topology`](@ref).

The covariance estimator is updated externally through
`RecurrentNetwork.estimators`, including while this rule is inactive.

By convention, the rule acts only on nonzero weights. Entries whose scale is
zero are also skipped.
"""
struct RatePlasticityScaledCovariance <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  B::Float64
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
        w_new = w_old + effective_learning_rate * scale * covariance_now[i,j]
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
      w_new = w_old + effective_learning_rate * scale * (covariance_now[i,j] + B * rates_post[i] * rates_pre[j])
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
        w_new = w_old + effective_learning_rate * scale * covariance_now[j,i]
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
      w_new = w_old + effective_learning_rate * scale * (covariance_now[j,i] + B * rates_post[i] * rates_pre[j])
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
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end


"""
    RatePlasticityCovarianceQuadraticallyStabilized
    RatePlasticityCovarianceQuadraticallyStabilized(pop_post, synapses_post_pre, pop_pre, Δt, α1, α2, learning_rate, covariance_estimator; is_active=Ref(true), w_min=1E-8, w_max=Inf)

Covariance-based plasticity rule with an additional stabilizing term that
scales quadratically with the weight.
the rule is w <- w + Δt * learning_rate * (α1 * C_post_pre(t) + α2 * w^2)
With an update every Δt seconds. `learning_rate` is interpreted per unit time,
so each update is internally scaled by the elapsed plasticity interval.

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
      w_new = w_old + effective_learning_rate * (α1 * covariance_now[i,j] + α2 * w_old^2)
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
      w_new = w_old + effective_learning_rate * (α1 * covariance_now[j,i] + α2 * w_old^2)
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
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_quadratic_covariance_plasticity!(
    weights,
    covariance_estimator.covariance_now,
    α1,
    α2,
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
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64)
  return _update_quadratic_covariance_plasticity_transposed!(
    weights,
    covariance_estimator.covariance_estimator.covariance_now,
    α1,
    α2,
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
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

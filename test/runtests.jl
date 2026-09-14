using PlasticRecurrentNeuralNetworks
using LinearAlgebra
using Random
using Statistics
using Test

include("rate_inputs.jl")

@testset "PlasticRecurrentNeuralNetworks.jl" begin
  PNN = PlasticRecurrentNeuralNetworks

  @testset "Excitatory self connection fixed points" begin
    dt = 1e-3
    t_end = 8.0
    tau = 0.1
    mu = 20.0
    n = 1
    initial_rate = 5.0
    self_connection_weights = (0.0,0.345,0.8)

    for w_self in self_connection_weights
      expected_fixed_point = mu / (1.0 - w_self)
      @test expected_fixed_point > 0.0

      population = PNN.LinearRateNeuralPopulation(
        PNN.ExcitatoryRateNeuron(tau;rate_saturation=100.0),
        n,
        initial_rates=initial_rate,
      )
      input = PNN.RateFixedInput(n,mu)
      synapse = PNN.RateLinearSynapses(fill(w_self,(n,n)))
      rec = PNN.RCRate(population,t_end,0.05)
      network = PNN.RecurrentNetwork(
        populations=(population,),
        connections=((population,input),(population,synapse,population)),
        recorders=(rec,),
      )

      t_now = 0.0
      while t_now <= t_end
        t_now = PNN.dynamic_step!(t_now,dt,network)
      end

      content = PNN.get_content(rec)
      @test content.times[end] >= t_end - rec.dt - dt
      @test isapprox(content.rates[end,1],expected_fixed_point; rtol=1e-3)
      @test isapprox(population.rates_now[1],expected_fixed_point; rtol=1e-3)
    end
  end

  @testset "Two-dimensional E/I fixed point" begin
    dt = 1e-3
    t_end = 30.0
    tau_exc = 90e-3
    tau_inh = 66e-3
    dt_recorder = 50e-3
    mu = 20.0
    n = 1
    initial_rates = [5.0,5.0]
    w_matrix_form = [[0.3,2.0] [-1.0,-2.0]]
    expected_fixed_point = (I - w_matrix_form) \ fill(mu,2)

    @test isapprox(expected_fixed_point,[9.756097560975611,13.170731707317074])

    e_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(tau_exc;rate_saturation=100.0),
      n,
      initial_rates=initial_rates[1],
    )
    i_population = PNN.LinearRateNeuralPopulation(
      PNN.InhibitoryRateNeuron(tau_inh;rate_saturation=100.0),
      n,
      initial_rates=initial_rates[2],
    )

    e_input = PNN.RateFixedInput(n,mu)
    i_input = PNN.RateFixedInput(n,mu)
    rec_e = PNN.RCRate(e_population,t_end,dt_recorder)
    rec_i = PNN.RCRate(i_population,t_end,dt_recorder)

    scalar_weight_matrix(x::Number) = fill(abs(float(x)),n,n)

    network = PNN.RecurrentNetwork(
      populations=(e_population,i_population),
      connections=(
        (e_population,e_input),
        (i_population,i_input),
        (e_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[1,1])),e_population),
        (i_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[2,1])),e_population),
        (e_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[1,2])),i_population),
        (i_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[2,2])),i_population),
      ),
      recorders=(rec_e,rec_i),
    )

    t_now = 0.0
    while t_now <= t_end
      t_now = PNN.dynamic_step!(t_now,dt,network)
    end

    content_e = PNN.get_content(rec_e)
    content_i = PNN.get_content(rec_i)
    final_rates = [content_e.rates[end,1],content_i.rates[end,1]]

    @test isapprox(final_rates,expected_fixed_point; rtol=1e-12)
    @test isapprox(
      [e_population.rates_now[1],i_population.rates_now[1]],
      expected_fixed_point;
      rtol=1e-12,
    )
  end

  @testset "LinearRateNeuralPopulation rate saturation" begin
    neuron_type = PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0)
    inhibitory_type = PNN.InhibitoryRateNeuron(1.0;rate_saturation=100.0)

    @test neuron_type.rate_saturation == 100.0
    @test inhibitory_type.rate_saturation == 100.0

    population = PNN.LinearRateNeuralPopulation(
      neuron_type,
      3,
      initial_rates=[90.0,20.0,5.0],
    )
    population.input_alloc .= [200.0,50.0,-10.0]

    @test PNN.local_update!(0.0,1.0,population) === nothing
    @test population.rates_now == [100.0,50.0,0.0]
  end

  @testset "RateLinearSynapses constructor" begin
    weights = [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    synapse = PNN.RateLinearSynapses(weights)

    @test synapse.n_post == 2
    @test synapse.n_pre == 3
    @test synapse.weights === weights
    @test !isdefined(PNN,:RateLinearSynapse)
    @test_throws MethodError PNN.RateLinearSynapses(3,2,weights)
  end

  @testset "RateLinearSynapses forward signal accumulation" begin
    weights = [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    synapse = PNN.RateLinearSynapses(weights)
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=0.0,
    )
    excitatory_pre = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[0.5,1.0,2.0],
    )
    inhibitory_pre = PNN.LinearRateNeuralPopulation(
      PNN.InhibitoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[0.5,1.0,2.0],
    )

    initial_input = [10.0,-3.0]
    expected_drive = weights * excitatory_pre.rates_now

    post_population.input_alloc .= initial_input
    @test PNN.forward_signal!(0.0,0.1,post_population,synapse,excitatory_pre) === nothing
    @test isapprox(post_population.input_alloc,initial_input + expected_drive)

    post_population.input_alloc .= initial_input
    @test PNN.forward_signal!(0.0,0.1,post_population,synapse,inhibitory_pre) === nothing
    @test isapprox(post_population.input_alloc,initial_input - expected_drive)
  end

  @testset "RCRate recorder" begin
    population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      [1.0,2.0,3.0],
      zeros(3),
      zeros(3),
    )

    rec = PNN.RCRate(population,0.5,0.2)
    @test length(rec.times) == 4
    @test size(rec.rates) == (4,3)
    @test rec.krec == 0

    @test PNN.record!(0.0,rec) === nothing
    @test rec.krec == 1
    @test rec.times[1] == 0.0
    @test rec.rates[1,:] == [1.0,2.0,3.0]

    population.rates_now .= [4.0,5.0,6.0]
    @test PNN.record!(0.1,rec) === nothing
    @test rec.krec == 1
    @test isnan(rec.times[2])

    @test PNN.record!(0.2,rec) === nothing
    @test rec.krec == 2
    @test rec.times[2] == 0.2
    @test rec.rates[2,:] == [4.0,5.0,6.0]

    population.rates_now .= [7.0,8.0,9.0]
    @test PNN.record!(0.39,rec) === nothing
    @test rec.krec == 2

    @test PNN.record!(0.4,rec) === nothing
    @test rec.krec == 3
    @test rec.times[3] == 0.4
    @test rec.rates[3,:] == [7.0,8.0,9.0]

    @test PNN.record!(0.6,rec) === nothing
    @test rec.krec == 3

    content = PNN.get_content(rec)
    @test content.times == [0.0,0.2,0.4]
    @test content.rates == [
      1.0 2.0 3.0
      4.0 5.0 6.0
      7.0 8.0 9.0
    ]

    PNN.reset!(rec)
    @test rec.krec == 0
    @test all(isnan,rec.times)
    @test all(isnan,rec.rates)

    rec_with_offset = PNN.RCRate(population,0.5,0.2;t_start=0.1)
    @test PNN.record!(0.0,rec_with_offset) === nothing
    @test rec_with_offset.krec == 0
    @test PNN.record!(0.1,rec_with_offset) === nothing
    @test rec_with_offset.krec == 1
    @test PNN.record!(0.3,rec_with_offset) === nothing
    @test rec_with_offset.krec == 2
    @test rec_with_offset.times[2] == 0.3
  end

  @testset "RCWeights recorder" begin
    weights = [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    rec = PNN.RCWeights(weights,0.5,0.2)

    @test rec.weight_matrix_now === weights
    @test length(rec.times) == 4
    @test size(rec.weights) == (4,2,3)
    @test rec.krec == 0

    @test PNN.record!(0.0,rec) === nothing
    @test rec.krec == 1
    @test rec.times[1] == 0.0
    @test rec.weights[1,:,:] == weights

    weights .= [
      7.0 8.0 9.0
      10.0 11.0 12.0
    ]
    @test PNN.record!(0.1,rec) === nothing
    @test rec.krec == 1
    @test isnan(rec.times[2])

    @test PNN.record!(0.2,rec) === nothing
    @test rec.krec == 2
    @test rec.times[2] == 0.2
    @test rec.weights[2,:,:] == weights

    weights .= [
      13.0 14.0 15.0
      16.0 17.0 18.0
    ]
    @test PNN.record!(0.4,rec) === nothing
    @test rec.krec == 3
    @test rec.times[3] == 0.4
    @test rec.weights[3,:,:] == weights

    content = PNN.get_content(rec)
    @test content.times == [0.0,0.2,0.4]
    @test size(content.weights) == (3,2,3)
    @test content.weights[1,:,:] == [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    @test content.weights[3,:,:] == [
      13.0 14.0 15.0
      16.0 17.0 18.0
    ]

    synapse = PNN.RateLinearSynapses(weights)
    rec_from_synapse = PNN.RCWeights(synapse,0.4,0.2)
    @test rec_from_synapse.weight_matrix_now === synapse.weights

    legacy_rec = PNN.RCWeights(weights,0.2,0.4,0.1)
    @test legacy_rec.dt == 0.2
    @test legacy_rec.t_end == 0.4
    @test legacy_rec.t_start == 0.1

    @test PNN.reset!(rec) === nothing
    @test rec.krec == 0
    @test all(isnan,rec.times)
    @test all(isnan,rec.weights)
  end

  @testset "RateMeanEstimator" begin
    mean_now = [1.0,2.0,3.0]
    rates_now = [10.0,20.0,30.0]
    propagation_factor = 0.75
    @test PNN._update_rate_mean!(mean_now,rates_now,propagation_factor) === nothing
    @test isapprox(
      mean_now,
      propagation_factor .* [1.0,2.0,3.0] .+
        (1.0 - propagation_factor) .* rates_now;
      rtol=1e-12,
    )

    population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[10.0,20.0],
    )
    tau_mean = 0.5
    dt_trace = 0.2
    dt_rate = 0.01
    estimator = PNN.RateMeanEstimator(population,tau_mean,dt_trace)
    decay = exp(-dt_trace / tau_mean)

    @test estimator.propagation_factor == decay
    @test estimator.mean_now == [0.0,0.0]
    @test estimator.t_last_update == -Inf

    @test PNN.local_update!(0.0,dt_rate,estimator) === nothing
    first_expected = (1 - decay) .* [10.0,20.0]
    @test isapprox(estimator.mean_now,first_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    population.rates_now .= [30.0,50.0]
    @test PNN.local_update!(0.1,dt_rate,estimator) === nothing
    @test isapprox(estimator.mean_now,first_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    @test PNN.local_update!(0.2,dt_rate,estimator) === nothing
    second_expected = decay .* first_expected .+ (1 - decay) .* [30.0,50.0]
    @test isapprox(estimator.mean_now,second_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.2

    @test PNN.reset!(estimator) === nothing
    @test estimator.mean_now == [0.0,0.0]
    @test estimator.t_last_update == -Inf
  end

  @testset "RateCovarianceEstimator" begin
    second_moment_now = [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    covariance_now = fill(NaN,2,3)
    post_rates = [10.0,20.0]
    pre_rates = [1.0,2.0,3.0]
    post_means = [3.0,4.0]
    pre_means = [0.5,1.0,1.5]
    propagation_factor = 0.75
    second_moment_initial = copy(second_moment_now)
    second_moment_expected = propagation_factor .* second_moment_initial .+
      (1.0 - propagation_factor) .* (post_rates * pre_rates')
    covariance_expected = second_moment_expected .- post_means * pre_means'

    @test PNN._update_rate_covariance!(
      second_moment_now,
      covariance_now,
      post_rates,
      pre_rates,
      post_means,
      pre_means,
      propagation_factor,
    ) === nothing
    @test isapprox(second_moment_now,second_moment_expected; rtol=1e-12)
    @test isapprox(covariance_now,covariance_expected; rtol=1e-12)

    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[10.0,20.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[1.0,2.0,3.0],
    )
    tau_covariance = 0.5
    dt_trace = 0.2
    dt_rate = 0.01
    post_mean_estimator = PNN.RateMeanEstimator(post_population,tau_covariance,dt_trace)
    pre_mean_estimator = PNN.RateMeanEstimator(pre_population,tau_covariance,dt_trace)
    estimator = PNN.RateCovarianceEstimator(post_mean_estimator,pre_mean_estimator)
    decay = exp(-dt_trace / tau_covariance)
    input_factor = 1 - decay

    @test estimator.pop_post === post_population
    @test estimator.pop_pre === pre_population
    @test estimator.τ == tau_covariance
    @test estimator.dt == dt_trace
    @test estimator.propagation_factor == decay
    @test estimator.mean_post_estimator === post_mean_estimator
    @test estimator.mean_pre_estimator === pre_mean_estimator
    @test size(estimator.second_moment_now) == (2,3)
    @test size(estimator.covariance_now) == (2,3)
    @test estimator.t_last_update == -Inf

    post_rates = copy(post_population.rates_now)
    pre_rates = copy(pre_population.rates_now)
    @test PNN.local_update!(0.0,dt_rate,post_mean_estimator) === nothing
    @test PNN.local_update!(0.0,dt_rate,pre_mean_estimator) === nothing
    @test PNN.local_update!(0.0,dt_rate,estimator) === nothing
    mean_post_expected = input_factor .* post_rates
    mean_pre_expected = input_factor .* pre_rates
    second_moment_expected = input_factor .* (post_rates * pre_rates')
    covariance_expected = second_moment_expected .- mean_post_expected * mean_pre_expected'

    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test isapprox(estimator.second_moment_now,second_moment_expected; rtol=1e-12)
    @test isapprox(estimator.covariance_now,covariance_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    post_population.rates_now .= [30.0,50.0]
    pre_population.rates_now .= [4.0,5.0,6.0]
    @test PNN.local_update!(0.1,dt_rate,post_mean_estimator) === nothing
    @test PNN.local_update!(0.1,dt_rate,pre_mean_estimator) === nothing
    @test PNN.local_update!(0.1,dt_rate,estimator) === nothing
    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test isapprox(estimator.second_moment_now,second_moment_expected; rtol=1e-12)
    @test isapprox(estimator.covariance_now,covariance_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    post_rates = copy(post_population.rates_now)
    pre_rates = copy(pre_population.rates_now)
    @test PNN.local_update!(0.2,dt_rate,post_mean_estimator) === nothing
    @test PNN.local_update!(0.2,dt_rate,pre_mean_estimator) === nothing
    @test PNN.local_update!(0.2,dt_rate,estimator) === nothing
    mean_post_expected = decay .* mean_post_expected .+ input_factor .* post_rates
    mean_pre_expected = decay .* mean_pre_expected .+ input_factor .* pre_rates
    second_moment_expected = decay .* second_moment_expected .+
      input_factor .* (post_rates * pre_rates')
    covariance_expected = second_moment_expected .- mean_post_expected * mean_pre_expected'

    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test isapprox(estimator.second_moment_now,second_moment_expected; rtol=1e-12)
    @test isapprox(estimator.covariance_now,covariance_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.2

    @test PNN.reset!(estimator) === nothing
    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test isapprox(
      estimator.second_moment_now,
      mean_post_expected * mean_pre_expected';
      rtol=1e-12,
    )
    @test all(iszero,estimator.covariance_now)
    @test estimator.t_last_update == -Inf

    initialized_post_mean = PNN.RateMeanEstimator(
      post_population,
      tau_covariance,
      dt_trace;
      initial_mean=copy(post_population.rates_now),
    )
    initialized_pre_mean = PNN.RateMeanEstimator(
      pre_population,
      tau_covariance,
      dt_trace;
      initial_mean=copy(pre_population.rates_now),
    )
    initialized_estimator = PNN.RateCovarianceEstimator(
      initialized_post_mean,
      initialized_pre_mean,
    )
    initialized_second_moment =
      initialized_post_mean.mean_now * initialized_pre_mean.mean_now'
    @test initialized_estimator.second_moment_now == initialized_second_moment
    @test all(iszero,initialized_estimator.covariance_now)

    @test PNN.local_update!(0.0,dt_rate,initialized_post_mean) === nothing
    @test PNN.local_update!(0.0,dt_rate,initialized_pre_mean) === nothing
    @test PNN.local_update!(0.0,dt_rate,initialized_estimator) === nothing
    @test isapprox(
      initialized_estimator.second_moment_now,
      initialized_second_moment;
      rtol=1e-12,
    )
    @test all(iszero,initialized_estimator.covariance_now)

    empty_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      0,
    )
    empty_mean = PNN.RateMeanEstimator(empty_population,tau_covariance,dt_trace)
    empty_estimator = PNN.RateCovarianceEstimator(empty_mean,empty_mean)
    @test size(empty_estimator.second_moment_now) == (0,0)
    @test size(empty_estimator.covariance_now) == (0,0)
    @test PNN.reset!(empty_estimator) === nothing

    explicit_time_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator;
      τ=1.0,
      dt=0.3,
    )
    @test explicit_time_estimator.τ == 1.0
    @test explicit_time_estimator.dt == 0.3
    @test explicit_time_estimator.propagation_factor == exp(-0.3 / 1.0)

    different_tau_mean_estimator = PNN.RateMeanEstimator(pre_population,0.7,dt_trace)
    @test_throws AssertionError PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_tau_mean_estimator,
    )
    @test PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_tau_mean_estimator;
      τ=tau_covariance,
    ).τ == tau_covariance

    different_dt_mean_estimator = PNN.RateMeanEstimator(pre_population,tau_covariance,0.3)
    @test_throws AssertionError PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_dt_mean_estimator,
    )
    @test PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_dt_mean_estimator;
      dt=dt_trace,
    ).dt == dt_trace

    mismatched_mean_estimator = PNN.RateMeanEstimator(pre_population,tau_covariance,dt_trace)
    @test_throws AssertionError PNN.RateCovarianceEstimator(
      post_population,
      pre_population,
      mismatched_mean_estimator,
      pre_mean_estimator,
      tau_covariance,
      dt_trace,
    )
  end

  @testset "Network estimator phase with inactive plasticity" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      1,
      initial_rates=[2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      1,
      initial_rates=[3.0],
    )
    post_mean = PNN.RateMeanEstimator(post_population,1.0,0.1)
    pre_mean = PNN.RateMeanEstimator(pre_population,1.0,0.1)
    covariance = PNN.RateCovarianceEstimator(post_mean,pre_mean)
    synapse = PNN.RateLinearSynapses(ones(1,1))
    rule = PNN.RatePlasticityCovariance(
      post_population,
      synapse,
      pre_population,
      0.1,
      0.2,
      covariance;
      is_active=Ref(false),
    )
    network = PNN.RecurrentNetwork(
      populations=(post_population,pre_population),
      estimators=(post_mean,pre_mean,covariance),
      plasticity_rules=(rule,),
    )

    @test PNN.RecurrentNetwork().estimators == ()
    @test network.estimators === (post_mean,pre_mean,covariance)
    @test PNN.dynamic_step!(0.0,0.01,network) == 0.01
    input_factor = 1.0 - exp(-0.1)
    @test isapprox(post_mean.mean_now,input_factor .* post_population.rates_now; rtol=1e-12)
    @test isapprox(pre_mean.mean_now,input_factor .* pre_population.rates_now; rtol=1e-12)
    expected_second_moment = input_factor .* (post_population.rates_now * pre_population.rates_now')
    expected_covariance = expected_second_moment .- post_mean.mean_now * pre_mean.mean_now'
    @test isapprox(covariance.second_moment_now,expected_second_moment; rtol=1e-12)
    @test isapprox(covariance.covariance_now,expected_covariance; rtol=1e-12)
    @test post_mean.t_last_update == 0.0
    @test pre_mean.t_last_update == 0.0
    @test covariance.t_last_update == 0.0
    @test synapse.weights == ones(1,1)
    @test rule.t_last_update[] == -Inf
  end

  @testset "Rate plasticity update kernels" begin
    homeostatic_weights = [
      0.0  1.0
      2.0  0.0
    ]
    @test PNN._update_homeostatic_scaling!(
      homeostatic_weights,
      [2.0,4.0],
      [1.0,5.0],
      3.0,
      0.1,
      0.1,
      10.0,
    ) === nothing
    @test isapprox(homeostatic_weights,[0.0 1.4; 0.4 0.0]; rtol=1e-12)

    covariance = [
      2.0  -1.0  -100.0
      0.5   1.5    -2.0
    ]
    covariance_weights = [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test PNN._update_covariance_plasticity!(
      covariance_weights,
      covariance,
      [0.5,1.0,1.5],
      [10.0,20.0],
      0.5,
      0.0, # α_leak
      0.02,
      0.1,
      3.01,
    ) === nothing
    @test isapprox(covariance_weights,[
      1.09  0.0   0.1
      2.11  3.01  3.01
    ]; rtol=1e-12)

    zero_B_weights = ones(2,3)
    @test PNN._update_covariance_plasticity!(
      zero_B_weights,
      covariance,
      fill(NaN,3),
      fill(NaN,2),
      0.0,
      0.0, # α_leak
      0.02,
      -Inf,
      Inf,
    ) === nothing
    @test isapprox(zero_B_weights,ones(2,3) .+ 0.02 .* covariance; rtol=1e-12)

    scaled_weights = [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    scale_matrix = [
      1.0  0.0  2.0
      0.5  3.0  0.25
    ]
    @test PNN._update_scaled_covariance_plasticity!(
      scaled_weights,
      covariance,
      scale_matrix,
      [0.5,1.0,1.5],
      [10.0,20.0],
      -0.5,
      0.0, # α_leak
      0.02,
      0.1,
      3.01,
    ) === nothing
    @test isapprox(scaled_weights,[
      0.99   0.0   0.1
      1.955  2.49  3.01
    ]; rtol=1e-12)

    zero_B_scaled_weights = ones(2,3)
    @test PNN._update_scaled_covariance_plasticity!(
      zero_B_scaled_weights,
      covariance,
      scale_matrix,
      fill(NaN,3),
      fill(NaN,2),
      -0.0,
      0.0, # α_leak
      0.02,
      -Inf,
      Inf,
    ) === nothing
    @test isapprox(
      zero_B_scaled_weights,
      ones(2,3) .+ 0.02 .* scale_matrix .* covariance;
      rtol=1e-12,
    )

    quadratic_weights = [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test PNN._update_quadratic_covariance_plasticity!(
      quadratic_weights,
      covariance,
      -0.5,
      0.25,
      0.0, # α_leak
      0.02,
      0.1,
      3.01,
    ) === nothing
    @test isapprox(quadratic_weights,[
      0.985  0.0   2.005
      2.015  3.01  3.01
    ]; rtol=1e-12)
  end

  @testset "RatePlasticityHomeostaticScaling" begin
    population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[2.0,4.0],
    )
    synapse = PNN.RateLinearSynapses([
      0.0  1.0
      2.0  0.0
    ])
    mean_estimator = PNN.RateMeanEstimator(
      population,
      1.0,
      0.1;
      initial_mean=[1.0,5.0],
    )
    active_ref = Ref(true)
    rule = PNN.RatePlasticityHomeostaticScaling(
      population,
      synapse,
      population,
      3.0,
      0.2,
      0.5,
      mean_estimator;
      w_min=0.1,
      w_max=10.0,
      is_active=active_ref,
    )

    @test !ismutabletype(typeof(rule))
    @test rule.is_active === active_ref
    @test rule.is_active[]
    initial_weights = copy(synapse.weights)
    @test PNN.plasticity_off!(rule) === nothing
    @test !rule.is_active[]
    @test PNN.plasticity!(0.2,0.01,rule) === nothing
    @test synapse.weights == initial_weights
    @test rule.t_last_update[] == -Inf
    @test PNN.plasticity_on!(rule) === nothing
    @test rule.is_active[]
    @test PNN.plasticity!(0.2,0.01,rule) === nothing
    @test synapse.weights != initial_weights
    @test rule.t_last_update[] == 0.2
    copyto!(synapse.weights,initial_weights)

    rule.t_last_update[] = 0.0
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights == [
      0.0  1.0
      2.0  0.0
    ]
    @test rule.t_last_update[] == 0.0

    @test PNN.plasticity!(0.2,0.01,rule) === nothing
    @test isapprox(synapse.weights,[0.0 1.4; 0.4 0.0]; rtol=1e-12)
    @test rule.t_last_update[] == 0.2

    synapse_small_dt = PNN.RateLinearSynapses([
      0.0  1.0
      2.0  0.0
    ])
    rule_small_dt = PNN.RatePlasticityHomeostaticScaling(
      population,
      synapse_small_dt,
      population,
      3.0,
      0.1,
      0.5,
      mean_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    @test rule_small_dt.is_active[]
    rule_small_dt.t_last_update[] = 0.0
    @test PNN.plasticity!(0.1,0.01,rule_small_dt) === nothing
    @test isapprox(synapse_small_dt.weights,[0.0 1.2; 1.2 0.0]; rtol=1e-12)
    @test isapprox(
      synapse.weights[1,2] - 1.0,
      2 * (synapse_small_dt.weights[1,2] - 1.0);
      rtol=1e-12,
    )

    other_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=0.0,
    )
    other_estimator = PNN.RateMeanEstimator(other_population,1.0,0.1)
    @test_throws AssertionError PNN.RatePlasticityHomeostaticScaling(
      population,
      synapse,
      population,
      3.0,
      0.2,
      0.5,
      other_estimator,
    )
  end

  @testset "RatePlasticityCovariance" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[1.0,2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[3.0,4.0,5.0],
    )
    post_mean_estimator = PNN.RateMeanEstimator(
      post_population,
      1.0,
      0.1;
      initial_mean=[10.0,20.0],
    )
    pre_mean_estimator = PNN.RateMeanEstimator(
      pre_population,
      1.0,
      0.1;
      initial_mean=[0.5,1.0,1.5],
    )
    covariance_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator,
    )
    covariance_estimator.covariance_now .= [
      2.0  -1.0  -100.0
      0.5   1.5    -2.0
    ]
    synapse = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    active_ref = Ref(true)
    rule = PNN.RatePlasticityCovariance(
      post_population,
      synapse,
      pre_population,
      0.5,
      0.1,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=3.01,
      is_active=active_ref,
    )

    @test rule.pop_pre === pre_population
    @test rule.pop_post === post_population
    @test rule.synapses_post_pre === synapse
    @test rule.covariance_estimator === covariance_estimator
    @test rule.B == 0.5
    @test !ismutabletype(typeof(rule))
    @test rule.is_active === active_ref
    @test rule.t_last_update[] == -Inf
    initial_weights = copy(synapse.weights)
    @test PNN.plasticity_off!(rule) === nothing
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights == initial_weights
    @test rule.t_last_update[] == -Inf
    @test PNN.plasticity_on!(rule) === nothing
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights != initial_weights
    @test rule.t_last_update[] == 0.1
    copyto!(synapse.weights,initial_weights)

    plain_rule = PNN.RatePlasticityCovariance(
      post_population,
      synapse,
      pre_population,
      0.1,
      0.2,
      covariance_estimator;
      is_active=active_ref,
    )
    @test plain_rule.B == 0.0
    @test plain_rule.is_active[]
    @test plain_rule.is_active === rule.is_active
    PNN.plasticity_off!(plain_rule)
    @test !rule.is_active[]
    PNN.plasticity_on!(rule)

    rule.t_last_update[] = 0.0
    @test PNN.plasticity!(0.05,0.01,rule) === nothing
    @test synapse.weights == [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test rule.t_last_update[] == 0.0

    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test isapprox(synapse.weights,[
      1.09  0.0   0.1
      2.11  3.01  3.01
    ]; rtol=1e-12)
    @test rule.t_last_update[] == 0.1

    synapse_small_dt = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule_small_dt = PNN.RatePlasticityCovariance(
      post_population,
      synapse_small_dt,
      pre_population,
      0.5,
      0.05,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    @test rule_small_dt.is_active[]
    rule_small_dt.t_last_update[] = 0.0
    @test PNN.plasticity!(0.05,0.01,rule_small_dt) === nothing
    @test isapprox(
      synapse.weights[1,1] - 1.0,
      2 * (synapse_small_dt.weights[1,1] - 1.0);
      rtol=1e-12,
    )

    transposed_synapse = PNN.RateLinearSynapses(ones(3,2))
    transposed_rule = PNN.RatePlasticityCovariance(
      pre_population,
      transposed_synapse,
      post_population,
      1.0,
      0.1,
      1.0,
      PNN.CovarianceTransposed(covariance_estimator);
      w_min=-Inf,
      w_max=Inf,
    )
    @test PNN.plasticity!(0.0,0.01,transposed_rule) === nothing
    @test isapprox(transposed_synapse.weights,[
       1.7   2.05
       1.9   3.15
      -7.5   3.8
    ]; rtol=1e-12)

    swapped_covariance_estimator = PNN.RateCovarianceEstimator(
      pre_mean_estimator,
      post_mean_estimator,
    )
    @test_throws AssertionError PNN.RatePlasticityCovariance(
      post_population,
      synapse,
      pre_population,
      0.5,
      0.1,
      0.2,
      swapped_covariance_estimator,
    )

    wrong_size_synapse = PNN.RateLinearSynapses(ones(2,2))
    @test_throws AssertionError PNN.RatePlasticityCovariance(
      post_population,
      wrong_size_synapse,
      pre_population,
      0.5,
      0.1,
      0.2,
      covariance_estimator,
    )
  end

  @testset "RatePlasticityScaledCovariance" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[1.0,2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[3.0,4.0,5.0],
    )
    post_mean_estimator = PNN.RateMeanEstimator(
      post_population,
      1.0,
      0.1;
      initial_mean=[10.0,20.0],
    )
    pre_mean_estimator = PNN.RateMeanEstimator(
      pre_population,
      1.0,
      0.1;
      initial_mean=[0.5,1.0,1.5],
    )
    covariance_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator,
    )
    covariance_estimator.covariance_now .= [
      2.0  -1.0  -100.0
      0.5   1.5    -2.0
    ]
    scale_matrix = [
      1.0  0.0  2.0
      0.5  3.0  0.25
    ]
    synapse = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    active_ref = Ref(true)
    rule = PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      scale_matrix,
      -0.5,
      0.1,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=3.01,
      is_active=active_ref,
    )

    @test rule.pop_pre === pre_population
    @test rule.pop_post === post_population
    @test rule.synapses_post_pre === synapse
    @test rule.covariance_estimator === covariance_estimator
    @test rule.scale_matrix === scale_matrix
    @test rule.B == -0.5
    @test !ismutabletype(typeof(rule))
    @test rule.is_active === active_ref
    @test rule.t_last_update[] == -Inf
    initial_weights = copy(synapse.weights)
    PNN.plasticity_off!(rule)
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights == initial_weights
    @test rule.t_last_update[] == -Inf
    PNN.plasticity_on!(rule)
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights != initial_weights
    @test rule.t_last_update[] == 0.1
    copyto!(synapse.weights,initial_weights)

    plain_rule = PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      scale_matrix,
      0.1,
      0.2,
      covariance_estimator,
    )
    @test plain_rule.B == 0.0
    @test plain_rule.is_active[]

    rule.t_last_update[] = 0.0
    @test PNN.plasticity!(0.05,0.01,rule) === nothing
    @test synapse.weights == [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test rule.t_last_update[] == 0.0

    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test isapprox(synapse.weights,[
      0.99   0.0   0.1
      1.955  2.49  3.01
    ]; rtol=1e-12)
    @test rule.t_last_update[] == 0.1

    synapse_small_dt = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule_small_dt = PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse_small_dt,
      pre_population,
      scale_matrix,
      -0.5,
      0.05,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    rule_small_dt.t_last_update[] = 0.0
    @test PNN.plasticity!(0.05,0.01,rule_small_dt) === nothing
    @test isapprox(
      synapse.weights[1,1] - 1.0,
      2 * (synapse_small_dt.weights[1,1] - 1.0);
      rtol=1e-12,
    )

    transposed_scale = [
       1.0  2.0
       0.5  0.0
      -1.0  0.25
    ]
    transposed_synapse = PNN.RateLinearSynapses(ones(3,2))
    transposed_rule = PNN.RatePlasticityScaledCovariance(
      pre_population,
      transposed_synapse,
      post_population,
      transposed_scale,
      1.0,
      0.1,
      1.0,
      PNN.CovarianceTransposed(covariance_estimator);
      w_min=-Inf,
      w_max=Inf,
    )
    @test PNN.plasticity!(0.0,0.01,transposed_rule) === nothing
    @test isapprox(transposed_synapse.weights,[
      1.7   3.1
      1.45  1.0
      9.5   1.7
    ]; rtol=1e-12)

    swapped_covariance_estimator = PNN.RateCovarianceEstimator(
      pre_mean_estimator,
      post_mean_estimator,
    )
    @test_throws AssertionError PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      scale_matrix,
      0.5,
      0.1,
      0.2,
      swapped_covariance_estimator,
    )

    wrong_size_synapse = PNN.RateLinearSynapses(ones(2,2))
    @test_throws AssertionError PNN.RatePlasticityScaledCovariance(
      post_population,
      wrong_size_synapse,
      pre_population,
      scale_matrix,
      0.5,
      0.1,
      0.2,
      covariance_estimator,
    )

    @test_throws AssertionError PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      ones(2,2),
      0.5,
      0.1,
      0.2,
      covariance_estimator,
    )
  end

  @testset "Signed-square-root covariance plasticity" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),2,
      initial_rates=[1.0,2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),3,
      initial_rates=[3.0,4.0,5.0],
    )
    post_mean = PNN.RateMeanEstimator(post_population,1.0,0.1;initial_mean=[2.0,3.0])
    pre_mean = PNN.RateMeanEstimator(pre_population,1.0,0.1;initial_mean=[5.0,7.0,11.0])
    covariance = PNN.RateCovarianceEstimator(post_mean,pre_mean)
    covariance.covariance_now .= [4.0 -9.0 0.0; 16.0 -25.0 36.0]

    synapse = PNN.RateLinearSynapses([1.0 0.0 2.0; 3.0 4.0 5.0])
    active = Ref(true)
    rule = PNN.RatePlasticitySQRC(
      post_population,synapse,pre_population,0.5,0.2,0.5,covariance;
      α_leak=0.25,w_min=-Inf,w_max=Inf,is_active=active,
    )
    @test rule.pop_pre === pre_population
    @test rule.pop_post === post_population
    @test rule.synapses_post_pre === synapse
    @test rule.covariance_estimator === covariance
    @test rule.B == 0.5
    @test rule.α_leak == 0.25
    @test rule.is_active === active
    @test !ismutabletype(typeof(rule))
    initial_weights = copy(synapse.weights)
    @test PNN.plasticity_off!(rule) === nothing
    @test PNN.plasticity!(0.0,0.01,rule) === nothing
    @test synapse.weights == initial_weights
    @test rule.t_last_update[] == -Inf
    PNN.plasticity_on!(rule)
    @test PNN.plasticity!(0.0,0.01,rule) === nothing
    expected_weights = [1.675 0.0 3.05; 4.075 4.45 7.125]
    @test isapprox(synapse.weights,expected_weights;rtol=1e-12)
    weights_after_update = copy(synapse.weights)
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights == weights_after_update

    plain_synapse = PNN.RateLinearSynapses(ones(2,3))
    plain_rule = PNN.RatePlasticitySQRC(
      post_population,plain_synapse,pre_population,1.0,1.0,covariance;
      w_min=-Inf,w_max=Inf,
    )
    @test plain_rule.B == 0.0
    PNN.plasticity!(0.0,0.01,plain_rule)
    @test plain_synapse.weights == [3.0 -2.0 1.0; 5.0 -4.0 7.0]

    transposed_synapse = PNN.RateLinearSynapses(ones(3,2))
    transposed_rule = PNN.RatePlasticitySQRC(
      pre_population,transposed_synapse,post_population,1.0,1.0,
      PNN.CovarianceTransposed(covariance);w_min=-Inf,w_max=Inf,
    )
    PNN.plasticity!(0.0,0.01,transposed_rule)
    @test transposed_synapse.weights == [3.0 5.0; -2.0 -4.0; 1.0 7.0]

    scale = [1.0 0.0 2.0; -1.0 0.5 0.25]
    scaled_synapse = PNN.RateLinearSynapses(ones(2,3))
    scaled_rule = PNN.RatePlasticityScaledSQRC(
      post_population,scaled_synapse,pre_population,scale,1.0,1.0,covariance;
      w_min=-Inf,w_max=Inf,
    )
    @test scaled_rule.B == 0.0
    @test scaled_rule.scale_matrix === scale
    @test PNN.plasticity!(0.0,0.01,scaled_rule) === nothing
    @test scaled_synapse.weights == [3.0 1.0 1.0; -3.0 -1.5 2.5]

    transposed_scale = [1.0 2.0; 0.5 0.0; -1.0 0.25]
    transposed_scaled_synapse = PNN.RateLinearSynapses(ones(3,2))
    transposed_scaled_rule = PNN.RatePlasticityScaledSQRC(
      pre_population,transposed_scaled_synapse,post_population,
      transposed_scale,1.0,1.0,PNN.CovarianceTransposed(covariance);
      w_min=-Inf,w_max=Inf,
    )
    PNN.plasticity!(0.0,0.01,transposed_scaled_rule)
    @test transposed_scaled_synapse.weights == [3.0 9.0; -0.5 1.0; 1.0 2.5]

    scaled_with_b = PNN.RatePlasticityScaledSQRC(
      post_population,PNN.RateLinearSynapses(ones(2,3)),pre_population,scale,
      0.5,1.0,1.0,covariance,
    )
    @test scaled_with_b.B == 0.5

    wrong_synapse = PNN.RateLinearSynapses(ones(2,2))
    @test_throws AssertionError PNN.RatePlasticitySQRC(
      post_population,wrong_synapse,pre_population,1.0,1.0,covariance,
    )
    @test_throws AssertionError PNN.RatePlasticityScaledSQRC(
      post_population,synapse,pre_population,ones(2,2),1.0,1.0,covariance,
    )
  end

  @testset "Covariance weight leakage" begin
    post = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0), 2,
    )
    pre = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0), 3,
    )
    post_mean = PNN.RateMeanEstimator(post,1.0,0.1;initial_mean=[2.0,3.0])
    pre_mean = PNN.RateMeanEstimator(pre,1.0,0.1;initial_mean=[1.0,2.0,4.0])
    covariance = PNN.RateCovarianceEstimator(post_mean,pre_mean)
    C = [0.5 -1.0 2.0; 1.5 0.25 -0.5]
    initial = [1.0 2.0 0.0; 3.0 4.0 5.0]
    scales = [0.5 0.0 2.0; 1.0 2.0 -0.5]

    @testset "scaled=$scaled transposed=$transposed B=$B" for
        scaled in (false,true), transposed in (false,true), B in (0.0,0.5)
      covariance.covariance_now .= C
      post_mean.mean_now .= [2.0,3.0]
      pre_mean.mean_now .= [1.0,2.0,4.0]
      pop_post,pop_pre = transposed ? (pre,post) : (post,pre)
      estimator = transposed ? PNN.CovarianceTransposed(covariance) : covariance
      weights = transposed ? permutedims(initial) : copy(initial)
      A = scaled ? (transposed ? permutedims(scales) : copy(scales)) : ones(size(weights))
      synapse = PNN.RateLinearSynapses(copy(weights))
      constructor = scaled ? PNN.RatePlasticityScaledCovariance : PNN.RatePlasticityCovariance
      args = scaled ? (pop_post,synapse,pop_pre,A) : (pop_post,synapse,pop_pre)
      # Exercise both the implicit-B constructor and the explicit-B constructor.
      args = B == 0.0 ? (args...,0.1,0.2,estimator) : (args...,B,0.1,0.2,estimator)
      default_rule = constructor(args...;w_min=-Inf)
      zero_rule = constructor(args...;α_leak=0.0,w_min=-Inf)
      @test default_rule.α_leak == zero_rule.α_leak == 0.0
      PNN.plasticity!(0.0,0.01,default_rule)
      default_result = copy(synapse.weights)
      synapse.weights .= weights
      PNN.plasticity!(0.0,0.01,zero_rule)
      @test isapprox(synapse.weights,default_result;rtol=1e-12,atol=1e-12)

      synapse.weights .= weights
      rule = constructor(args...;α_leak=0.75,w_min=-Inf)
      @test rule.α_leak == 0.75
      PNN.plasticity_off!(rule)
      @test PNN.plasticity!(0.0,0.01,rule) === nothing
      @test synapse.weights == weights
      @test rule.t_last_update[] == -Inf
      PNN.plasticity_on!(rule)
      rule.t_last_update[] = 0.0
      @test PNN.plasticity!(0.05,0.01,rule) === nothing
      @test synapse.weights == weights
      @test rule.t_last_update[] == 0.0

      drive = C + B * ([2.0,3.0] * [1.0,2.0,4.0]')
      drive = transposed ? permutedims(drive) : drive
      expected = weights + 0.02 .* A .* (drive - 0.75 .* weights)
      expected[weights .== 0.0] .= 0.0
      @test PNN.plasticity!(0.1,0.01,rule) === nothing
      @test isapprox(synapse.weights,expected;rtol=1e-12,atol=1e-12)
      @test rule.t_last_update[] == 0.1

      # With no covariance or mean drive, only scaled weight decay remains.
      covariance.covariance_now .= 0.0
      post_mean.mean_now .= 0.0
      pre_mean.mean_now .= 0.0
      synapse.weights .= weights
      @test PNN.plasticity!(0.3,0.01,rule) === nothing
      @test isapprox(synapse.weights,weights .* (1.0 .- 0.015 .* A);
        rtol=1e-12,atol=1e-12)

      # Clamp the full update, while preserving zero weights and masked entries.
      synapse.weights .= weights
      bounded_rule = constructor(args...;α_leak=100.0,w_min=0.1,w_max=4.5)
      expected = clamp.(weights .* (1.0 .- 2.0 .* A),0.1,4.5)
      skipped = (weights .== 0.0) .| (A .== 0.0)
      expected[skipped] .= weights[skipped]
      @test PNN.plasticity!(0.0,0.01,bounded_rule) === nothing
      @test isapprox(synapse.weights,expected;rtol=1e-12,atol=1e-12)
    end
  end

  @testset "RatePlasticityCovarianceQuadraticallyStabilized" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[1.0,2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[3.0,4.0,5.0],
    )
    post_mean_estimator = PNN.RateMeanEstimator(post_population,1.0,0.1)
    pre_mean_estimator = PNN.RateMeanEstimator(pre_population,1.0,0.1)
    covariance_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator,
    )
    covariance_estimator.covariance_now .= [
      2.0  -1.0  -100.0
      0.5   1.5    -2.0
    ]
    synapse = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    active_ref = Ref(true)
    rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      synapse,
      pre_population,
      0.1,
      -0.5,
      0.25,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=3.01,
      is_active=active_ref,
    )

    @test rule.pop_pre === pre_population
    @test rule.pop_post === post_population
    @test rule.synapses_post_pre === synapse
    @test rule.covariance_estimator === covariance_estimator
    @test rule.α1 == -0.5
    @test rule.α2 == 0.25
    @test !ismutabletype(typeof(rule))
    @test rule.is_active === active_ref
    @test rule.t_last_update[] == -Inf
    initial_weights = copy(synapse.weights)
    PNN.plasticity_off!(rule)
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights == initial_weights
    @test rule.t_last_update[] == -Inf
    PNN.plasticity_on!(rule)
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights != initial_weights
    @test rule.t_last_update[] == 0.1
    copyto!(synapse.weights,initial_weights)

    rule.t_last_update[] = 0.0
    @test PNN.plasticity!(0.05,0.01,rule) === nothing
    @test synapse.weights == [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test rule.t_last_update[] == 0.0

    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test isapprox(synapse.weights,[
      0.985  0.0   2.005
      2.015  3.01  3.01
    ]; rtol=1e-12)
    @test rule.t_last_update[] == 0.1

    synapse_small_dt = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule_small_dt = PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      synapse_small_dt,
      pre_population,
      0.05,
      -0.5,
      0.25,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    @test rule_small_dt.is_active[]
    rule_small_dt.t_last_update[] = 0.0
    @test PNN.plasticity!(0.05,0.01,rule_small_dt) === nothing
    @test isapprox(
      synapse.weights[1,1] - 1.0,
      2 * (synapse_small_dt.weights[1,1] - 1.0);
      rtol=1e-12,
    )

    @test rule.α_leak == 0.0
    @testset "Leakage transposed=$transposed" for transposed in (false,true)
      weights = transposed ? permutedims(initial_weights) : copy(initial_weights)
      C = transposed ? permutedims(covariance_estimator.covariance_now) :
        copy(covariance_estimator.covariance_now)
      pop_post,pop_pre = transposed ? (pre_population,post_population) :
        (post_population,pre_population)
      estimator = transposed ? PNN.CovarianceTransposed(covariance_estimator) :
        covariance_estimator
      leak_synapse = PNN.RateLinearSynapses(copy(weights))
      args = (pop_post,leak_synapse,pop_pre,0.1,-0.5,0.25,0.2,estimator)
      default_rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(args...;w_min=-Inf)
      zero_rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(args...;α_leak=0.0,w_min=-Inf)
      @test zero_rule.α_leak == default_rule.α_leak == 0.0
      PNN.plasticity!(0.0,0.01,default_rule)
      default_result = copy(leak_synapse.weights)
      leak_synapse.weights .= weights
      PNN.plasticity!(0.0,0.01,zero_rule)
      @test isapprox(leak_synapse.weights,default_result;rtol=1e-12,atol=1e-12)

      leak_synapse.weights .= weights
      leak_rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(args...;α_leak=0.75,w_min=-Inf)
      @test leak_rule.α_leak == 0.75
      PNN.plasticity_off!(leak_rule)
      @test PNN.plasticity!(0.0,0.01,leak_rule) === nothing
      @test leak_synapse.weights == weights
      @test leak_rule.t_last_update[] == -Inf
      PNN.plasticity_on!(leak_rule)
      leak_rule.t_last_update[] = 0.0
      @test PNN.plasticity!(0.05,0.01,leak_rule) === nothing
      @test leak_synapse.weights == weights
      @test leak_rule.t_last_update[] == 0.0
      expected = weights + 0.02 .* (-0.5 .* C + 0.25 .* weights.^2 - 0.75 .* weights)
      expected[weights .== 0.0] .= 0.0
      @test PNN.plasticity!(0.1,0.01,leak_rule) === nothing
      @test isapprox(leak_synapse.weights,expected;rtol=1e-12,atol=1e-12)
      @test leak_rule.t_last_update[] == 0.1

      leak_synapse.weights .= weights
      decay_rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(
        pop_post,leak_synapse,pop_pre,0.05,0.0,0.0,0.2,estimator;
        α_leak=0.75,w_min=-Inf,
      )
      @test PNN.plasticity!(0.0,0.01,decay_rule) === nothing
      @test isapprox(leak_synapse.weights,0.9925 .* weights;rtol=1e-12,atol=1e-12)

      leak_synapse.weights .= weights
      bounded_rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(
        args...;α_leak=100.0,w_min=0.1,w_max=1.5,
      )
      expected = clamp.(weights + 0.02 .* (-0.5 .* C + 0.25 .* weights.^2 - 100.0 .* weights),0.1,1.5)
      expected[weights .== 0.0] .= 0.0
      @test PNN.plasticity!(0.0,0.01,bounded_rule) === nothing
      @test isapprox(leak_synapse.weights,expected;rtol=1e-12,atol=1e-12)
    end

    swapped_covariance_estimator = PNN.RateCovarianceEstimator(
      pre_mean_estimator,
      post_mean_estimator,
    )
    @test_throws AssertionError PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      synapse,
      pre_population,
      0.1,
      -0.5,
      0.25,
      0.2,
      swapped_covariance_estimator,
    )

    wrong_size_synapse = PNN.RateLinearSynapses(ones(2,2))
    @test_throws AssertionError PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      wrong_size_synapse,
      pre_population,
      0.1,
      -0.5,
      0.25,
      0.2,
      covariance_estimator,
    )
  end

  @testset "Homogeneous weight matrices" begin
    weights = PNN.generate_homogeneous_weight_matrix(2,3,6.0)
    @test size(weights) == (2,3)
    @test weights == fill(2.0,2,3)
    @test isapprox(vec(sum(weights;dims=2)),fill(6.0,2);rtol=1e-12)

    no_self_weights = PNN.generate_homogeneous_noselfconnected_matrix(4,6.0)
    @test size(no_self_weights) == (4,4)
    @test diag(no_self_weights) == zeros(4)
    @test all(no_self_weights[.!Matrix{Bool}(I,4,4)] .== 2.0)
    @test isapprox(vec(sum(no_self_weights;dims=2)),fill(6.0,4);rtol=1e-12)
  end

  @testset "generate_ring_topology" begin
    locations = PNN.place_neurons_on_ring(4; offset=0.5)
    @test isapprox(locations,0.5 .+ [0.0,π / 2,π,3π / 2]; rtol=1e-12)
    wrapped_locations = PNN.place_neurons_on_ring(4; offset=3π / 2)
    @test all(0.0 .<= wrapped_locations .< 2π)
    @test isapprox(wrapped_locations,[3π / 2,0.0,π / 2,π]; rtol=1e-12, atol=1e-12)
    negative_offset_locations = PNN.place_neurons_on_ring(4; offset=-π / 2)
    @test all(0.0 .<= negative_offset_locations .< 2π)
    @test isapprox(negative_offset_locations,[3π / 2,0.0,π / 2,π]; rtol=1e-12, atol=1e-12)

    scale_factor = 2.5
    kappa = 3.0
    locations_post, locations_pre, topology = PNN.generate_ring_topology(
      4,
      4;
      scale_factor=scale_factor,
      kappa=kappa,
    )

    @test isapprox(locations_post,[0.0,π / 2,π,3π / 2]; rtol=1e-12)
    @test isapprox(locations_pre,[0.0,π / 2,π,3π / 2]; rtol=1e-12)
    @test size(topology) == (4,4)
    @test all(isapprox.(diag(topology),scale_factor; rtol=1e-12))
    @test isapprox(topology[1,2],scale_factor * exp(-kappa); rtol=1e-12)
    @test isapprox(topology[1,4],scale_factor * exp(-kappa); rtol=1e-12)
    @test isapprox(topology[1,3],scale_factor * exp(-2 * kappa); rtol=1e-12)

    _, _, rectangular_topology = PNN.generate_ring_topology(
      2,
      3;
      scale_factor=scale_factor,
      kappa=kappa,
    )
    @test size(rectangular_topology) == (2,3)
    @test isapprox(maximum(rectangular_topology),scale_factor; rtol=1e-12)
  end
end

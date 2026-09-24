function t1_update_allocations(pop)
  PlasticRecurrentNeuralNetworks.local_update!(0.0,0.5,pop)
  return @allocated PlasticRecurrentNeuralNetworks.local_update!(0.0,0.5,pop)
end

@testset "Quadratic T1 rate populations" begin
  PNN = PlasticRecurrentNeuralNetworks
  population_types = (PNN.LinearRateNeuralPopulation,PNN.QuadraticT1RateNeuralPopulation)

  @testset "Construction and cleanup" begin
    for Population in population_types, Neuron in (PNN.ExcitatoryRateNeuron,PNN.InhibitoryRateNeuron)
      neuron = Neuron(2.0;rate_saturation=100.0)
      pop = Population(neuron,3)
      @test pop isa PNN.RateNeuralPopulation
      @test pop.neuron_type === neuron
      @test pop.n == 3
      @test pop.rates_now == zeros(3)
      @test pop.input_alloc == zeros(3)
      @test pop.utility_alloc == zeros(3)
      @test Population(neuron,3;initial_rates=2.0).rates_now == fill(2.0,3)
      rates = [1.0,2.0,3.0]
      @test Population(neuron,3;initial_rates=rates).rates_now === rates
      @test_throws AssertionError Population(neuron,2;initial_rates=rates)
      input = ones(3)
      utility = fill(7.0,3)
      raw = Population(neuron,3,rates,input,utility)
      @test raw.rates_now === rates
      @test raw.input_alloc === input
      @test raw.utility_alloc === utility
      @test PNN.clean_up!(raw) === nothing
      @test input == zeros(3)
      @test rates == [1.0,2.0,3.0]
      @test utility == fill(7.0,3)
      empty_pop = Population(neuron,0)
      @test PNN.local_update!(0.0,0.25,empty_pop) === nothing
      @test PNN.clean_up!(empty_pop) === nothing
      @test isempty(empty_pop.rates_now)
    end
  end

  @testset "Quadratic T1 Euler dynamics" begin
    for Neuron in (PNN.ExcitatoryRateNeuron,PNN.InhibitoryRateNeuron)
      pop = PNN.QuadraticT1RateNeuralPopulation(
        Neuron(2.0;rate_saturation=10.0),5;initial_rates=[4.0,4.0,4.0,9.0,0.0],
      )
      pop.input_alloc .= [3.0,0.0,-3.0,20.0,0.5]
      pop.utility_alloc .= 7.0
      @test PNN.local_update!(0.0,0.5,pop) === nothing
      @test isapprox(pop.rates_now,[5.25,3.0,3.0,10.0,0.0625];atol=1e-12,rtol=1e-12)
      @test pop.input_alloc == [3.0,0.0,-3.0,20.0,0.5]
      @test pop.utility_alloc == fill(7.0,5)
      # An oversize Euler step can cross zero; retain the existing lower clamp.
      pop.input_alloc .= -2.0
      @test PNN.local_update!(0.5,4.0,pop) === nothing
      @test pop.rates_now == zeros(5)
      @test t1_update_allocations(pop) == 0
    end
  end

  @testset "Input compatibility" begin
    neuron = PNN.ExcitatoryRateNeuron(1.0)
    quadratic = PNN.QuadraticT1RateNeuralPopulation(neuron,2)
    linear = PNN.LinearRateNeuralPopulation(neuron,2)
    inputs = (
      PNN.RateFixedInput([2.0,-1.0]),
      PNN.RateNoisyHomogeneousInput(2,2.0,0.5),
      PNN.RateSimpleCorrelatedInput(2,2.0,0.5,0.4),
      PNN.RateGeneralCorrelatedInput([1.0 0.4;0.4 1.0],2.0,0.5),
    )
    for input in inputs
      for pop in (linear,quadratic)
        pop.input_alloc .= 1.0
        Random.seed!(31415)
        @test PNN.forward_signal!(0.0,0.25,pop,input) === nothing
      end
      @test isapprox(quadratic.input_alloc,linear.input_alloc;atol=1e-12,rtol=1e-12)
      quadratic.rates_now .= 0.0
      expected = 0.25 .* max.(quadratic.input_alloc,0.0).^2
      PNN.local_update!(0.0,0.25,quadratic)
      @test isapprox(quadratic.rates_now,expected;atol=1e-12,rtol=1e-12)
      wrong_size = PNN.QuadraticT1RateNeuralPopulation(neuron,1)
      @test_throws DimensionMismatch PNN.forward_signal!(0.0,0.25,wrong_size,input)
    end
  end

  @testset "Mixed network update ordering" begin
    post = PNN.QuadraticT1RateNeuralPopulation(PNN.ExcitatoryRateNeuron(1.0),2;initial_rates=[4.0,4.0])
    pre = PNN.LinearRateNeuralPopulation(PNN.InhibitoryRateNeuron(1.0),1;initial_rates=2.0)
    synapse = PNN.RateLinearSynapses(ones(2,1))
    post_mean = PNN.RateMeanEstimator(post,1.0,0.25)
    pre_mean = PNN.RateMeanEstimator(pre,1.0,0.25)
    covariance = PNN.RateCovarianceEstimator(post_mean,pre_mean)
    rule = PNN.RatePlasticityCovariance(post,synapse,pre,0.25,0.1,covariance)
    rates = PNN.RCRate(post,0.5,0.25)
    weights = PNN.RCWeights(synapse,0.5,0.25)
    network = PNN.RecurrentNetwork(
      populations=(post,pre),
      connections=((post,PNN.RateFixedInput([5.0,1.0])),(post,synapse,pre)),
      estimators=(post_mean,pre_mean,covariance),plasticity_rules=(rule,),recorders=(rates,weights),
    )
    post.input_alloc .= 999.0
    @test PNN.dynamic_step!(0.0,0.25,network) == 0.25
    @test isapprox(post.input_alloc,[3.0,-1.0];atol=1e-12,rtol=1e-12)
    @test isapprox(post.rates_now,[5.25,3.0];atol=1e-12,rtol=1e-12)
    @test isapprox(pre.rates_now,[1.5];atol=1e-12,rtol=1e-12)
    a = exp(-0.25)
    expected_covariance = a*(1-a) .* ([5.25,3.0] * [1.5]')
    @test isapprox(post_mean.mean_now,(1-a).*[5.25,3.0];atol=1e-12,rtol=1e-12)
    @test isapprox(pre_mean.mean_now,(1-a).*[1.5];atol=1e-12,rtol=1e-12)
    @test isapprox(covariance.covariance_now,expected_covariance;atol=1e-12,rtol=1e-12)
    expected_weights = ones(2,1) .+ 0.025 .* expected_covariance
    @test isapprox(synapse.weights,expected_weights;atol=1e-12,rtol=1e-12)
    @test isapprox(PNN.get_content(rates).rates[1,:],[5.25,3.0];atol=1e-12,rtol=1e-12)
    @test isapprox(PNN.get_content(weights).weights[1,:,:],expected_weights;atol=1e-12,rtol=1e-12)
  end
end

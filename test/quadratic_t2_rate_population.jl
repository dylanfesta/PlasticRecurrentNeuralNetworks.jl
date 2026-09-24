function t2_update_allocations(pop)
  PlasticRecurrentNeuralNetworks.local_update!(0.0,0.5,pop)
  return @allocated PlasticRecurrentNeuralNetworks.local_update!(0.0,0.5,pop)
end

function t2_refresh_allocations(pop)
  PlasticRecurrentNeuralNetworks.update_rates!(pop)
  return @allocated PlasticRecurrentNeuralNetworks.update_rates!(pop)
end

@testset "Quadratic T2 rate populations" begin
  PNN = PlasticRecurrentNeuralNetworks
  Population = PNN.QuadraticT2RateNeuralPopulation

  @testset "Construction, refresh, and cleanup" begin
    for Neuron in (PNN.ExcitatoryRateNeuron,PNN.InhibitoryRateNeuron)
      neuron = Neuron(2.0;rate_saturation=9.0)
      pop = Population(neuron,3)
      @test pop isa PNN.RateNeuralPopulation
      @test pop.neuron_type === neuron
      @test pop.n == 3
      @test pop.voltages_now == pop.rates_now == zeros(3)
      @test pop.voltages_now !== pop.rates_now
      @test pop.input_alloc == pop.utility_alloc == zeros(3)
      @test Population(neuron,3;initial_voltages=-2.0).voltages_now == fill(-2.0,3)
      @test Population(neuron,3;initial_rates=4.0).voltages_now == fill(2.0,3)
      voltages = [-2.0,2.0,5.0]
      pop = Population(neuron,3;initial_voltages=voltages)
      @test pop.voltages_now === voltages
      @test pop.rates_now == [0.0,4.0,9.0]
      rates = [0.0,4.0,9.0]
      initialized = Population(neuron,3;initial_rates=rates)
      @test initialized.voltages_now == [0.0,2.0,3.0]
      @test initialized.rates_now == rates
      @test initialized.rates_now !== rates
      @test initialized.voltages_now !== rates
      pop.input_alloc .= 5.0
      pop.utility_alloc .= 7.0
      voltages .= [-4.0,0.5,10.0]
      @test PNN.update_rates!(pop) === nothing
      @test pop.rates_now == [0.0,0.25,9.0]
      @test voltages == [-4.0,0.5,10.0]
      @test pop.input_alloc == fill(5.0,3)
      @test pop.utility_alloc == fill(7.0,3)
      @test PNN.clean_up!(pop) === nothing
      @test pop.input_alloc == zeros(3)
      @test pop.utility_alloc == fill(7.0,3)
      @test pop.voltages_now == [-4.0,0.5,10.0]
      @test pop.rates_now == [0.0,0.25,9.0]
      @test t2_refresh_allocations(pop) == 0
      for kwargs in ((;),(;initial_voltages=Float64[]),(;initial_rates=Float64[]))
        empty_pop = Population(neuron,0;kwargs...)
        @test PNN.local_update!(0.0,0.25,empty_pop) === nothing
        @test PNN.update_rates!(empty_pop) === nothing
        @test PNN.clean_up!(empty_pop) === nothing
        @test isempty(empty_pop.voltages_now) && isempty(empty_pop.rates_now)
      end
      @test_throws ArgumentError Population(neuron,-1)
      @test_throws ArgumentError Population(neuron,3;initial_rates=1.0,initial_voltages=1.0)
      @test_throws DimensionMismatch Population(neuron,2;initial_rates=rates)
      @test_throws DimensionMismatch Population(neuron,2;initial_voltages=voltages)
      for invalid in (-1.0,10.0,NaN,Inf)
        @test_throws ArgumentError Population(neuron,3;initial_rates=invalid)
        @test_throws ArgumentError Population(neuron,0;initial_rates=invalid)
        @test_throws ArgumentError Population(neuron,1;initial_rates=[invalid])
      end
      for invalid in (-1.0,NaN,-Inf)
        @test_throws ArgumentError Population(Neuron(1.0;rate_saturation=invalid),1)
      end
      zero_cap = Population(Neuron(1.0;rate_saturation=0.0),1;initial_voltages=10.0)
      @test zero_cap.rates_now == [0.0]
      @test zero_cap.voltages_now == [10.0]
    end
  end

  @testset "Voltage integration and rate saturation" begin
    for Neuron in (PNN.ExcitatoryRateNeuron,PNN.InhibitoryRateNeuron)
      pop = Population(Neuron(2.0;rate_saturation=9.0),5;
        initial_voltages=[-4.0,-1.0,2.0,10.0,0.0])
      pop.input_alloc .= [0.0,7.0,-10.0,30.0,2.0]
      pop.utility_alloc .= 7.0
      @test PNN.local_update!(0.0,0.5,pop) === nothing
      @test isapprox(pop.voltages_now,[-3.0,1.0,-1.0,15.0,0.5];atol=1e-12,rtol=1e-12)
      @test isapprox(pop.rates_now,[0.0,1.0,0.0,9.0,0.25];atol=1e-12,rtol=1e-12)
      @test pop.input_alloc == [0.0,7.0,-10.0,30.0,2.0]
      @test pop.utility_alloc == fill(7.0,5)
      pop.input_alloc .= 0.0
      PNN.local_update!(0.5,1.0,pop)
      @test isapprox(pop.voltages_now,[-1.5,0.5,-0.5,7.5,0.25];atol=1e-12,rtol=1e-12)
      @test isapprox(pop.rates_now,[0.0,0.25,0.0,9.0,0.0625];atol=1e-12,rtol=1e-12)
      @test t2_update_allocations(pop) == 0
      uncapped = Population(Neuron(1.0;rate_saturation=Inf),1;initial_voltages=20.0)
      @test uncapped.rates_now == [400.0]
      uncapped.input_alloc .= 40.0
      PNN.local_update!(0.0,0.5,uncapped)
      @test uncapped.voltages_now == [30.0]
      @test uncapped.rates_now == [900.0]
    end
  end

  @testset "External inputs" begin
    neuron = PNN.ExcitatoryRateNeuron(1.0;rate_saturation=Inf)
    for input in (PNN.RateFixedInput([2.0,-1.0]),
        PNN.RateNoisyHomogeneousInput(2,2.0,0.5),
        PNN.RateSimpleCorrelatedInput(2,2.0,0.5,0.4),
        PNN.RateGeneralCorrelatedInput([1.0 0.4;0.4 1.0],2.0,0.5))
      linear = PNN.LinearRateNeuralPopulation(neuron,2)
      pop = Population(neuron,2)
      for target in (linear,pop)
        Random.seed!(31415)
        @test PNN.forward_signal!(0.0,0.25,target,input) === nothing
      end
      @test isapprox(pop.input_alloc,linear.input_alloc;atol=1e-12,rtol=1e-12)
      PNN.local_update!(0.0,0.25,pop)
      @test isapprox(pop.voltages_now,0.25 .* linear.input_alloc;atol=1e-12,rtol=1e-12)
      @test isapprox(pop.rates_now,max.(0.0,0.25 .* linear.input_alloc).^2;atol=1e-12,rtol=1e-12)
      @test_throws DimensionMismatch PNN.forward_signal!(0.0,0.25,Population(neuron,1),input)
    end
  end

  @testset "Multistep mixed network: rates throughout" begin
    post = Population(PNN.ExcitatoryRateNeuron(1.0;rate_saturation=9.0),2;initial_voltages=[-2.0,5.0])
    pre = PNN.LinearRateNeuralPopulation(PNN.InhibitoryRateNeuron(1.0),1;initial_rates=2.0)
    incoming = PNN.RateLinearSynapses(reshape([0.5,1.0],2,1))
    outgoing = PNN.RateLinearSynapses([0.2 0.3])
    post_mean = PNN.RateMeanEstimator(post,1.0,0.25)
    pre_mean = PNN.RateMeanEstimator(pre,1.0,0.25)
    covariance = PNN.RateCovarianceEstimator(post_mean,pre_mean)
    self_covariance = PNN.RateCovarianceEstimator(post_mean,post_mean)
    rule = PNN.RatePlasticityHomeostaticScaling(post,incoming,pre,1.0,1,0.25,0.1,post_mean)
    rec = PNN.RCRate(post,0.75,0.25)
    network = PNN.RecurrentNetwork(populations=(post,pre),
      connections=((post,PNN.RateFixedInput([8.0,-1.0])),(post,incoming,pre),(pre,outgoing,post)),
      estimators=(post_mean,pre_mean,covariance,self_covariance),plasticity_rules=(rule,),recorders=(rec,))
    v = [-2.0,5.0]
    r = [0.0,9.0]
    rpre = [2.0]
    w = reshape([0.5,1.0],2,1)
    mpost,mpre = zeros(2),zeros(1)
    qcross,qself = zeros(2,1),zeros(2,2)
    a = exp(-0.25)
    for (k,t) in enumerate((0.0,0.25,0.5,0.75))
      drive_post = [8.0,-1.0] - w * rpre
      drive_pre = [0.2 0.3] * r
      v += 0.25 .* (-v + drive_post)
      r = min.(9.0,max.(0.0,v).^2)
      rpre = clamp.(rpre + 0.25 .* (-rpre + drive_pre),0.0,200.0)
      mpost = a .* mpost + (1-a) .* r
      mpre = a .* mpre + (1-a) .* rpre
      qcross = a .* qcross + (1-a) .* (r * rpre')
      qself = a .* qself + (1-a) .* (r * r')
      w = max.(1e-8,w + 0.025 .* w .* r .* (mpost .- 1.0))
      post.input_alloc .= 999.0
      @test PNN.dynamic_step!(t,0.25,network) == t + 0.25
      @test isapprox(post.input_alloc,drive_post;atol=1e-12,rtol=1e-12)
      @test isapprox(pre.input_alloc,drive_pre;atol=1e-12,rtol=1e-12)
      @test isapprox(post.voltages_now,v;atol=1e-12,rtol=1e-12)
      @test isapprox(post.rates_now,r;atol=1e-12,rtol=1e-12)
      @test isapprox(pre.rates_now,rpre;atol=1e-12,rtol=1e-12)
      @test isapprox(post_mean.mean_now,mpost;atol=1e-12,rtol=1e-12)
      @test isapprox(pre_mean.mean_now,mpre;atol=1e-12,rtol=1e-12)
      @test isapprox(covariance.covariance_now,qcross - mpost * mpre';atol=1e-12,rtol=1e-12)
      @test isapprox(diag(self_covariance.covariance_now),diag(qself) - mpost.^2;atol=1e-12,rtol=1e-12)
      @test isapprox(incoming.weights,w;atol=1e-12,rtol=1e-12)
      @test isapprox(PNN.get_content(rec).rates[k,:],r;atol=1e-12,rtol=1e-12)
    end
  end
end

@testset "Rate population compatibility" begin
  PNN = PlasticRecurrentNeuralNetworks
  shared_population_types = (PNN.LinearRateNeuralPopulation,PNN.QuadraticT1RateNeuralPopulation,PNN.QuadraticT2RateNeuralPopulation)
  @testset "Shared components: post=$Post pre=$Pre" for Post in shared_population_types, Pre in shared_population_types
    post = Post(PNN.ExcitatoryRateNeuron(1.0),2;initial_rates=[2.0,3.0])
    for Neuron in (PNN.ExcitatoryRateNeuron,PNN.InhibitoryRateNeuron)
      pre = Pre(Neuron(1.0),3;initial_rates=[1.0,2.0,4.0])
      weights = [1.0 2.0 3.0;4.0 5.0 6.0]
      synapse = PNN.RateLinearSynapses(weights)
      post.input_alloc .= 2.0
      sign = Neuron === PNN.ExcitatoryRateNeuron ? 1.0 : -1.0
      @test PNN.forward_signal!(0.0,0.25,post,synapse,pre) === nothing
      @test isapprox(post.input_alloc,2.0 .+ sign .* (weights * [1.0,2.0,4.0]);atol=1e-12,rtol=1e-12)
    end

    pre = Pre(PNN.ExcitatoryRateNeuron(1.0),3;initial_rates=[1.0,2.0,4.0])
    post_mean = PNN.RateMeanEstimator(post,1.0,0.25)
    pre_mean = PNN.RateMeanEstimator(pre,1.0,0.25)
    covariance = PNN.RateCovarianceEstimator(post_mean,pre_mean)
    for estimator in (post_mean,pre_mean,covariance)
      @test PNN.local_update!(0.0,0.25,estimator) === nothing
    end
    a = exp(-0.25)
    @test isapprox(post_mean.mean_now,(1-a).*[2.0,3.0];atol=1e-12,rtol=1e-12)
    @test isapprox(pre_mean.mean_now,(1-a).*[1.0,2.0,4.0];atol=1e-12,rtol=1e-12)
    expected_covariance = a*(1-a) .* ([2.0,3.0] * [1.0,2.0,4.0]')
    @test isapprox(covariance.covariance_now,expected_covariance;atol=1e-12,rtol=1e-12)

    # Distinct positive and negative entries expose orientation/sign mistakes.
    covariance.covariance_now .= [4.0 -9.0 16.0;-25.0 36.0 -49.0]
    for transposed in (false,true)
      est = transposed ? PNN.CovarianceTransposed(covariance) : covariance
      target,source = transposed ? (pre,post) : (post,pre)
      mean_target = transposed ? pre_mean : post_mean
      C = transposed ? copy(covariance.covariance_now') : copy(covariance.covariance_now)
      @test PNN.covariance_pop_post(est) === target
      @test PNN.covariance_pop_pre(est) === source
      @test PNN.covariance_n_post(est) == target.n
      @test PNN.covariance_n_pre(est) == source.n
      @test PNN.covariance_at(est,1,2) == C[1,2]
      scale = fill(2.0,size(C))
      for kind in (:homeostatic,:covariance,:scaled_covariance,:sqrc,:scaled_sqrc,:stabilized)
        synapse = PNN.RateLinearSynapses(ones(size(C)))
        if kind === :homeostatic
          rule = PNN.RatePlasticityHomeostaticScaling(target,synapse,source,1.0,-1,0.25,0.1,mean_target;w_min=-Inf)
          expected = ones(size(C)) .- 0.025 .* target.rates_now .* (mean_target.mean_now .- 1.0)
        elseif kind === :covariance
          rule = PNN.RatePlasticityCovariance(target,synapse,source,0.25,0.1,est;w_min=-Inf)
          expected = ones(size(C)) .+ 0.025 .* C
        elseif kind === :scaled_covariance
          rule = PNN.RatePlasticityScaledCovariance(target,synapse,source,scale,0.25,0.1,est;w_min=-Inf)
          expected = ones(size(C)) .+ 0.025 .* scale .* C
        elseif kind === :sqrc
          rule = PNN.RatePlasticitySQRC(target,synapse,source,0.25,0.1,est;w_min=-Inf)
          expected = ones(size(C)) .+ 0.025 .* sign.(C) .* sqrt.(abs.(C))
        elseif kind === :scaled_sqrc
          rule = PNN.RatePlasticityScaledSQRC(target,synapse,source,scale,0.25,0.1,est;w_min=-Inf)
          expected = ones(size(C)) .+ 0.025 .* scale .* sign.(C) .* sqrt.(abs.(C))
        else
          rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(target,synapse,source,0.25,0.5,-0.2,0.1,est;w_min=-Inf)
          expected = ones(size(C)) .+ 0.025 .* (0.5 .* C .- 0.2)
        end
        @test PNN.plasticity!(0.0,0.25,rule) === nothing
        @test isapprox(synapse.weights,expected;atol=1e-12,rtol=1e-12)
        rec = PNN.RCWeights(synapse,0.5,0.25)
        @test PNN.record!(0.0,rec) === nothing
        @test isapprox(PNN.get_content(rec).weights[1,:,:],expected;atol=1e-12,rtol=1e-12)
      end
    end

    rec = PNN.RCRate(post,0.5,0.25)
    for t in (-0.25,0.0,0.1,0.25,0.5,0.75)
      @test PNN.record!(t,rec) === nothing
    end
    content = PNN.get_content(rec)
    @test content.times == [0.0,0.25,0.5]
    @test content.rates == repeat(post.rates_now',3,1)
    @test PNN.reset!(rec) === nothing
    @test isempty(PNN.get_content(rec).times)
  end

end

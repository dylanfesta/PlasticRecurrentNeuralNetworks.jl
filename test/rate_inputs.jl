PNNRateInputs = PlasticRecurrentNeuralNetworks

function isolated_rate_samples(input;
    dt::Float64=0.05,
    τ::Float64=0.1,
    steps::Int64=30_000,
    warmup_steps::Int64=1_000,
    seed::Int64=1234)
  Random.seed!(seed)
  population = PNNRateInputs.LinearRateNeuralPopulation(
    PNNRateInputs.ExcitatoryRateNeuron(τ;rate_saturation=100.0),
    input.n,
    initial_rates=50.0,
  )
  samples = Matrix{Float64}(undef,steps - warmup_steps,input.n)

  for step in 1:steps
    PNNRateInputs.clean_up!(population)
    PNNRateInputs.forward_signal!(0.0,dt,population,input)
    PNNRateInputs.local_update!(0.0,dt,population)
    if step > warmup_steps
      samples[step - warmup_steps,:] .= population.rates_now
    end
  end
  return samples
end

@testset "Rate inputs" begin
  @testset "Fixed input" begin
    input_values = [1.0,2.0,3.0]
    vector_input = PNNRateInputs.RateFixedInput(input_values)
    homogeneous_input = PNNRateInputs.RateFixedInput(3,2.5)
    population = PNNRateInputs.LinearRateNeuralPopulation(
      PNNRateInputs.ExcitatoryRateNeuron(0.1),3,
    )

    @test vector_input.n == 3
    @test vector_input.input === input_values
    @test homogeneous_input.input == fill(2.5,3)
    @test isnothing(PNNRateInputs.forward_signal!(0.0,0.01,population,vector_input))
    @test population.input_alloc == input_values
    @test_throws ArgumentError PNNRateInputs.RateFixedInput(-1,2.0)
    @test_throws DimensionMismatch PNNRateInputs.forward_signal!(
      0.0,0.01,population,PNNRateInputs.RateFixedInput(2,1.0),
    )
  end

  @testset "Noise calibration and validation" begin
    population = PNNRateInputs.LinearRateNeuralPopulation(
      PNNRateInputs.ExcitatoryRateNeuron(0.1),2,
    )
    input = PNNRateInputs.RateNoisyHomogeneousInput(2,50.0,3.0)
    expected_scale = 3.0 * sqrt(2.0 * 0.1 / 0.05 - 1.0)
    @test PNNRateInputs._rate_noise_scale(population,0.05,3.0) == expected_scale
    @test isapprox(PNNRateInputs.noise_scale(input,0.05,population),expected_scale)
    @test_throws ArgumentError PNNRateInputs.noise_scale(input,0.0,population)
    @test_throws ArgumentError PNNRateInputs.noise_scale(input,0.2,population)
    @test_throws ArgumentError PNNRateInputs._rate_noise_scale(population,0.0,3.0)
    @test_throws ArgumentError PNNRateInputs._rate_noise_scale(population,0.2,3.0)
    @test_throws ArgumentError PNNRateInputs.RateNoisyHomogeneousInput(-1,50.0,2.0)
    @test_throws ArgumentError PNNRateInputs.RateNoisyHomogeneousInput(2,50.0,-1.0)

    samples = isolated_rate_samples(
      PNNRateInputs.RateNoisyHomogeneousInput(4,50.0,2.0),
    )
    @test isapprox(mean(samples),50.0;atol=0.05)
    @test isapprox(std(samples),2.0;rtol=0.03)
    sample_correlations = cor(samples)
    @test maximum(abs,sample_correlations - I) < 0.04
  end

  @testset "Simple correlated input" begin
    @test_throws ArgumentError PNNRateInputs.RateSimpleCorrelatedInput(2,50.0,2.0,-0.1)
    @test_throws ArgumentError PNNRateInputs.RateSimpleCorrelatedInput(2,50.0,2.0,1.1)

    independent_samples = isolated_rate_samples(
      PNNRateInputs.RateSimpleCorrelatedInput(3,50.0,2.0,0.0);
      seed=4321,
    )
    @test maximum(abs,cor(independent_samples) - I) < 0.04

    θ = 0.4
    correlated_samples = isolated_rate_samples(
      PNNRateInputs.RateSimpleCorrelatedInput(3,50.0,2.0,θ);
      seed=5678,
    )
    expected_correlations = fill(θ,3,3)
    expected_correlations[diagind(expected_correlations)] .= 1.0
    @test isapprox(mean(correlated_samples),50.0;atol=0.08)
    @test isapprox(std(correlated_samples),2.0;rtol=0.04)
    @test maximum(abs,cor(correlated_samples) - expected_correlations) < 0.04

    shared_samples = isolated_rate_samples(
      PNNRateInputs.RateSimpleCorrelatedInput(3,50.0,2.0,1.0);
      steps=2_000,
      warmup_steps=100,
      seed=8765,
    )
    @test shared_samples[:,1] == shared_samples[:,2]
    @test shared_samples[:,2] == shared_samples[:,3]
  end

  @testset "General correlated input" begin
    requested_correlations = [
      NaN 0.2 0.4
      0.2 -10.0 0.6
      0.4 0.6 Inf
    ]
    input = PNNRateInputs.RateGeneralCorrelatedInput(
      requested_correlations,50.0,2.0,
    )
    expected_correlations = [
      1.0 0.2 0.4
      0.2 1.0 0.6
      0.4 0.6 1.0
    ]
    @test input.correlation_levels == expected_correlations

    samples = isolated_rate_samples(input;seed=2468)
    @test isapprox(vec(mean(samples;dims=1)),fill(50.0,3);atol=0.1)
    @test isapprox(vec(std(samples;dims=1)),fill(2.0,3);rtol=0.04)
    @test maximum(abs,cor(samples) - expected_correlations) < 0.04

    fully_shared = PNNRateInputs.RateGeneralCorrelatedInput(ones(3,3),50.0,2.0)
    shared_samples = isolated_rate_samples(
      fully_shared;steps=2_000,warmup_steps=100,seed=1357,
    )
    @test isapprox(shared_samples[:,1],shared_samples[:,2];atol=1e-12,rtol=0.0)
    @test isapprox(shared_samples[:,2],shared_samples[:,3];atol=1e-12,rtol=0.0)

    empty_input = PNNRateInputs.RateGeneralCorrelatedInput(zeros(0,0),50.0,2.0)
    empty_population = PNNRateInputs.LinearRateNeuralPopulation(
      PNNRateInputs.ExcitatoryRateNeuron(0.1),0,
    )
    @test isnothing(PNNRateInputs.forward_signal!(0.0,0.05,empty_population,empty_input))

    @test_throws DimensionMismatch PNNRateInputs.RateGeneralCorrelatedInput(
      zeros(2,3),50.0,2.0,
    )
    @test_throws ArgumentError PNNRateInputs.RateGeneralCorrelatedInput(
      [1.0 0.2; 0.3 1.0],50.0,2.0,
    )
    @test_throws ArgumentError PNNRateInputs.RateGeneralCorrelatedInput(
      [1.0 1.1; 1.1 1.0],50.0,2.0,
    )
    @test_throws ArgumentError PNNRateInputs.RateGeneralCorrelatedInput(
      [1.0 NaN; NaN 1.0],50.0,2.0,
    )
    @test_throws ArgumentError PNNRateInputs.RateGeneralCorrelatedInput(
      [1.0 0.9 0.9; 0.9 1.0 0.1; 0.9 0.1 1.0],50.0,2.0,
    )

    mismatched_population = PNNRateInputs.LinearRateNeuralPopulation(
      PNNRateInputs.ExcitatoryRateNeuron(0.1),2,
    )
    @test_throws DimensionMismatch PNNRateInputs.forward_signal!(
      0.0,0.05,mismatched_population,input,
    )
  end
end

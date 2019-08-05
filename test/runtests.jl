using Test, MeshAdaptiveDirectSearch, Random

@testset "halton" begin
    import MeshAdaptiveDirectSearch: haltonnumber,
    normalized_halton_direction, scaledhouseholder
    # examples from Audet & Dennis 2006
    @test 2*haltonnumber(2, 1) - 1 == 0.
    @test haltonnumber(5, 6) ≈ 6/25
    @test haltonnumber(7, 7) ≈ 1/49
    h = MeshAdaptiveDirectSearch.HaltonIterator{4}()
    u = first(Iterators.drop(h, 9))
    @test u ≈ [5/16, 10/27, 2/25, 22/49]
    q = normalized_halton_direction(u, 3)
    @test q == [-1, -1, -2, 0]
    @test scaledhouseholder([-1, -2]) == [3 -4; -4 -3]
    # example from Abramson et al. 2009, p. 958
    u = first(Iterators.drop(MeshAdaptiveDirectSearch.HaltonIterator{4}(), 12))
    q = normalized_halton_direction(u, 6)
    @test scaledhouseholder(q) == [36 0 -18 -36;
                                   0 54 0 0;
                                   -18 0 36 -36;
                                   -36 0 -36 -18]
end

@testset "orthomads" begin
    import MeshAdaptiveDirectSearch: scaledhouseholder,
    normalized_halton_direction, LogMesh, ℓ, Δ, determine_t!, update!
    mesh = LogMesh()
    g = MeshAdaptiveDirectSearch.OrthoDirectionGenerator(4, reduction = NoReduction(), t0 = 7)
    # p. 958 from Abramson et al. 2009
    for (success, t, Δᵐ) in zip([1, 1, -1, -1, -1, -1, 1, -1, -1],
                                [7, 8, 9, 10, 7, 8, 9, 11, 9, 10],
                                [1, 1, 1, 1, 1, 1/4, 1/16, 1/4, 1/16, 1/64])
        l = ℓ(mesh)
        τ = determine_t!(g, l)
        @test t == τ
        @test Δ(mesh) == Δᵐ
        update!(mesh, success)
    end
end

@testset "readme" begin
    Random.seed!(12418)
    f(x) = (1 - exp(-sum(abs2, x))) * max(sum(abs2, x .- [30, 40]), sum(abs2, x .+ [30, 40]))
    noisyf(x) = f(x) + .1 * randn()

    res = minimize(LtMADS(2), f, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
    @test res.f < 1e-9
    res = minimize(LtMADS(2), f, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10], constraints = [x -> sum(x) < .5])
    @test res.f < 1e-9
    res = minimize(OrthoMADS(2), f, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
    @test res.f < 1e-9
    res = minimize(RobustLtMADS(2), noisyf, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
    @test res.f < 1e-9
    res = minimize(RobustOrthoMADS(2), noisyf, [-2.1, 1.7], lowerbound = [-10, -10], upperbound = [10, 10])
    @test res.f < 1e-9
end

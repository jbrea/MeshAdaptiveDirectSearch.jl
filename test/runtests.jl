using Test, MeshAdaptiveDirectSearch

@testset "halton" begin
    import MeshAdaptiveDirectSearch: haltonnumber,
    normalized_halton_direction, scaledhouseholder
    @test 2*haltonnumber(2, 1) - 1 == 0.
    @test haltonnumber(5, 6) ≈ 6/25
    @test haltonnumber(7, 7) ≈ 1/49
    h = MeshAdaptiveDirectSearch.HaltonIterator{4}()
    u = first(Iterators.drop(h, 9))
    @test u ≈ [5/16, 10/27, 2/25, 22/49]
    q = normalized_halton_direction(u, 3)
    @test q == [-1, -1, -2, 0]
    @test scaledhouseholder([-1, -2]) == [3 -4; -4 -3]
end

module MeshAdaptiveDirectSearch

using StaticArrays, Random, ElasticArrays, LinearAlgebra, Primes
export MADS, LtMADS, OrthoMADS, RobustMADS, RobustLtMADS, RobustOrthoMADS,
Silent, Progress, minimize

# TODO:
# * QrMADS
# * MADS(suc,neg) and variants

####
#### Meshes
####

# implements Eq. 2.1 from Audet & Dennis 2006
mutable struct Mesh
    τ::Float64
    w⁺::Int
    w⁻::Int
    Δᵐ::Float64
end
Mesh(; τ = 4., Δᵐ = 1., w⁺ = 1, w⁻ = -1) = Mesh(τ, w⁺, w⁻, Δᵐ)
function update!(m::Mesh, i)
    i == 0 && return m
    if i > 0
        m.Δᵐ == 1. && return m
        w = rand(0:m.w⁺)
    else
        w = rand(m.w⁻:-1)
    end
    m.Δᵐ *= m.τ^w
    m
end
Δ(m::Mesh) = m.Δᵐ
ℓ(m::Mesh) = round(Int, -log(m.τ, m.Δᵐ))
# implements Eq. 2.1 from Audet & Dennis 2006 for w⁺ = -w⁻ = 1
mutable struct LogMesh
    τ::Int
    neglogΔᵐ::Int
end
LogMesh(; τ = 4, Δᵐ = 1) = LogMesh(τ, Int(-log(τ, Δᵐ)))
function update!(m::LogMesh, i)
    i == 0 && return m
    if i > 0
        m.neglogΔᵐ == 0 && return m
        m.neglogΔᵐ -= 1
    else
        m.neglogΔᵐ += 1
    end
    m
end
Δ(m::LogMesh) = 1/m.τ^m.neglogΔᵐ
ℓ(m::LogMesh) = m.neglogΔᵐ

####
#### LT
####

# Implements generation of the direction b(l), box on p. 203 from Audet &  Dennis 2006
struct LTDirectionGenerator{N}
    b::Dict{Int, NamedTuple{(:i, :v), Tuple{Int, SVector{N,Int}}}}
end
LTDirectionGenerator(N) = LTDirectionGenerator{N}(Dict{Int, NamedTuple{(:i, :v), Tuple{Int, SVector{N,Int}}}}())
function b(d::LTDirectionGenerator{N}, l::Int) where N
    haskey(d.b, l) && return d.b[l]
    i = rand(1:N)
    d.b[l] = (i = i,
              v = SVector{N}([rand((-1,1)) * (k == i ? 2^l : (2^l - 1)) for k in 1:N]))
end
iterator(g::LTDirectionGenerator, l) = LTDirectionIterator(g, l)

# Implements generation of the positive bias, box on p. 204 from Audet & Dennis 2006
struct LTDirectionIterator{N,Np}
    l::Int
    i::Int
    b::SVector{N,Int}
    rowperm::SVector{Np,Int}
    colperm::SVector{Np,Int}
    sum::MVector{N,Int}
end
function LTDirectionIterator(generator::LTDirectionGenerator{N}, l) where N
    i, v = b(generator, l)
    LTDirectionIterator{N,N-1}(l, i, v,
                               SVector{N-1}(randperm(N-1)),
                               SVector{N-1}(randperm(N-1)),
                               zeros(Int, N))
end

function L(i, j, l)
    j > i && return 0
    j == i && return rand((-1, 1)) * 2^l
    return rand(-2^l + 1:2^l - 1)
end

import Base: iterate, length
length(::LTDirectionIterator{N,Np}) where {N, Np} = N + 1
function iterate(it::LTDirectionIterator{N,Np}, j = 1) where {N, Np}
    j == N + 2 && return nothing
    j == N + 1 && return -SVector(it.sum), j + 1
    j == 1 && (it.sum .*= 0)
    if j == it.i
        v = it.b
    else
        v = SVector{N}([i == it.i ? 0 : L(it.rowperm[i - (i > it.i)],
                               it.colperm[j - (j > it.i)], it.l)
                        for i in 1:N])
    end
    it.sum .+= v
    v, j + 1
end


####
#### MADS
####

# implements general MADS, p. 193 from Audet & Dennis, 2006
abstract type AbstractMADS end
"""
    struct MADS{Tmesh,Tsearch,Tpoll} <: AbstractMADS
        mesh::Tmesh
        search::Tsearch
        poll::Tpoll
"""
struct MADS{Tmesh,Tsearch,Tpoll} <: AbstractMADS
    mesh::Tmesh
    search::Tsearch
    poll::Tpoll
end
function MADS(N; search = NoSearch(),
                 poll = LTDirectionGenerator(N),
                 mesh = LogMesh())
    MADS(mesh, search, poll)
end
"""
     LtMADS(N; search = NoSearch(), mesh = LogMesh())

Returns a `MADS` object with `poll = LTDirectionGenerator(N)` where `N` is the dimenionality of the problem.
See Audet & Dennis (2006), section 4, LTMADS
"""
function LtMADS(N; search = NoSearch(), mesh = LogMesh())
    MADS(mesh, search, LTDirectionGenerator(N))
end
"""
     LtMADS(N; search = NoSearch(), mesh = LogMesh())

Returns a `MADS` object with `poll = OrthoDirectionGenerator(N)` where `N` is the dimenionality of the problem.
See Abramson et al. (2009), ORTHOMADS
"""
function OrthoMADS(N; search = NoSearch(),
                      mesh = LogMesh())
    MADS(mesh, search, OrthoDirectionGenerator(N))
end

# implements robust mads, from Audet et al. 2018
struct Cache
    x::ElasticArray{Float64,2,1}
    y::Vector{Float64}
    incumbents::Vector{Int}
end
Cache(N) = Cache(ElasticArray{Float64}(undef, N, 0), Float64[], Int[])
import Base.push!
push!(c::Cache, x, y) = begin append!(c.x, x); push!(c.y, y) end
"""
    struct RobustMADS{Tmesh,Tsearch,Tpoll,Tkernel} <: AbstractMADS
        mesh::Tmesh
        search::Tsearch
        poll::Tpoll
        kernel::Tkernel
        cache::Cache
        f::Vector{Float64}
        P::Vector{Float64}
"""
struct RobustMADS{Tmesh,Tsearch,Tpoll,Tkernel} <: AbstractMADS
    mesh::Tmesh
    search::Tsearch
    poll::Tpoll
    kernel::Tkernel
    cache::Cache
    f::Vector{Float64}
    P::Vector{Float64}
end
"""
    RobustMADS(N; search = NoSearch(), poll = LTDirectionGenerator(N),
                  mesh = LogMesh(), kernel = GaussKernel(1, 1), cache = Cache(N))

Returns a `RobustMADS` object where `N` is the dimenionality of the problem.
"""
function RobustMADS(N; search = NoSearch(), poll = LTDirectionGenerator(N),
                    mesh = LogMesh(), kernel = GaussKernel(1, 1), cache = Cache(N))
    RobustMADS(mesh, search, poll, kernel, cache, Float64[], Float64[])
end
"""
    RobustLtMADS(N; kwargs...)

Returns a `RobustMADS` object with `poll =  LTDirectionGenerator(N)` where `N` is the dimenionality of the problem.
"""
function RobustLtMADS(N; kwargs...)
    RobustMADS(N, poll = LTDirectionGenerator(N), kwargs...)
end
"""
    RobustOrthoMADS(N; kwargs...)

Returns a `RobustMADS` object with `poll =  OrthoDirectionGenerator(N)` where `N` is the dimenionality of the problem.
"""
function RobustOrthoMADS(N; kwargs...)
    RobustMADS(N, poll = OrthoDirectionGenerator(N), kwargs...)
end
mutable struct GaussKernel
    β::Float64
    σ²::Float64
end
(g::GaussKernel)(x, y) = g() * reshape(exp.(-sum((x .- y).^2, dims = 1)/(2*g.σ²)), :)
(g::GaussKernel)() = 1/sqrt(2*π*g.σ²)
update!kernel!(g::GaussKernel, mesh) = g.σ² = (g.β * Δ(mesh))^2
function isnewincumbent(m::RobustMADS, x, fx, oldfx)
    update!kernel!(m.kernel, m.mesh)
    if length(m.cache.x) > 0
        ψ = m.kernel(m.cache.x, x)
        m.f .*= m.P
        m.f .+= ψ * fx
        m.P .+= ψ
        m.f ./= m.P
        push!(m.P, sum(ψ) + m.kernel())
        push!(m.f, (dot(m.cache.y, ψ) + m.kernel() * fx)/m.P[end])
    else
        push!(m.P, m.kernel())
        push!(m.f, fx)
    end
    append!(m.cache.x, x)
    push!(m.cache.y, fx)
    i = argmin(m.f)
    success = i == length(m.f)
    cachesuccess = length(m.cache.incumbents) == 0 || i != m.cache.incumbents[end]
    success || cachesuccess && push!(m.cache.incumbents, i)
    success ? 1 : cachesuccess ? 0 : -1, m.cache.x[:, i], m.f[i]
end

# barrier
function isvalid(cs, x)
    for c in cs
        c(x) || return false
    end
    return true
end

# poll stage
poll(m, f, constraints, x, fx) = poll(m, iterator(m.poll, ℓ(m.mesh)), Δ(m.mesh), f, constraints, x, fx)
isnewincumbent(m::MADS, x, fx, oldfx) = fx < oldfx ? 1 : -1, x, fx
@inline function poll(m, it, Δᵐ, f, constraints, x, fx)
    for v in it
        newx = clamp!(x .+ Δᵐ * v, -1., 1.)
        isvalid(constraints, newx) || continue
        fnewx = f(newx)
#         @show fnewx
        success, newincumbent, fnewx = isnewincumbent(m, newx, fnewx, fx)
        success >= 0 && return (incumbent = newincumbent, fx = fnewx, hasimproved = success)
    end
    (incumbent = x, fx = fx, hasimproved = -1)
end

# search stage
struct NoSearch end
search(m, f, constraints, x, fx) = search(m.search, f, constraints, x, fx)
search(::NoSearch, f, constraints, x, fx) = (incumbent = x, fx = fx, hasimproved = 0)

# implements OthoMADS, Abramson et al. 2009
@inline function haltonnumber(base::Integer, index::Integer)::Float64
    res = 0.
    f = 1 / base
    i = index
    while i > 0
       res += f * (i % base)
       i = div(i, base)
       f = f / base
    end
    res
end
struct HaltonIterator{N} end
function iterate(it::HaltonIterator{N}, i = 1) where N
    [haltonnumber(prime(k), i) for k in 1:N], i + 1
end
function normalized_halton_direction(u, l)
    α = 2^(l/2)/sqrt(length(u)) - 1/2
    q = normalize(2 * u .- ones(length(u)))
    while norm(round.(α * q)) < 2^(l/2) # TODO: make more efficient
        α += .1
    end
    round.(Int, (α - .1) * q)
end
scaledhouseholder(q) = sum(q.^2) * I - 2 * q * q'
mutable struct OrthoDirectionGenerator{N}
    t₀::Int
    ℓmax::Int
    tmax::Int
end
OrthoDirectionGenerator(N; t0 = N) = OrthoDirectionGenerator{N}(t0, 0, 0)
iterator(g::OrthoDirectionGenerator, l) = OrthoDirectionIterator(g, l)
struct OrthoDirectionIterator{N}
    H::Matrix{Float64}
end
function OrthoDirectionIterator(g::OrthoDirectionGenerator{N}, ℓ) where N
    if ℓ > g.ℓmax
        g.ℓmax = ℓ
        ℓ > g.tmax && (g.tmax = ℓ)
        t = ℓ + g.t₀
    else
        g.tmax += 1
        t = g.tmax
    end
    u = first(iterate(HaltonIterator{N}(), t))
    q = normalized_halton_direction(u, ℓ)
    H = scaledhouseholder(q)
    OrthoDirectionIterator{N}(H)
end
length(::OrthoDirectionIterator{N}) where N = 2N
function iterate(it::OrthoDirectionIterator{N}, i = 1) where N
    i > 2N && return nothing
    i <= N && return @view(it.H[:, i]), i + 1
    return -@view(it.H[:, i - N]), i + 1
end


function standardtransformation(lowerbound, upperbound)
    d = upperbound - lowerbound
    (to = x -> @.((x + 1)/2 * (upperbound - lowerbound) + lowerbound),
     from = x -> @.(2 * (x - lowerbound) / (upperbound - lowerbound) - 1))
end

@enum Verbosity Silent Progress
@enum StoppingReason MaxIterations MinMeshSize
"""
    minimize(m, f, x0 = zeros(length(x0));
                   lowerbound = -ones(length(x0)),
                   upperbound = ones(length(x0)),
                   max_iterations = 10^4,
                   min_mesh_size = eps(Float64)/2,
                   constraints = [],
                   verbosity = Silent)

Minimize function `f` with method `m`.
For possible methods `m` see [`LtMADS`](@ref), [`OrthoMADS`](@ref), [`MADS`](@ref),
[`RobustLtMADS`](@ref), [`RobustOrthoMADS`](@ref), [`RobustMADS`](@ref).
Constraints can be defined by boolean functions, e.g.
`constraints = [x -> sum(x) > 1, x -> x[1]^2 < 3]`.
Possible `verbosity`-levels are `Silent` or `Progress`.
"""
function minimize(m::AbstractMADS, f, x0 = zeros(length(x0));
                  lowerbound = -ones(length(x0)),
                  upperbound = ones(length(x0)),
                  max_iterations = 10^4,
                  min_mesh_size = eps(Float64)/2,
                  verbosity = Silent,
                  constraints = [])
    isvalid(constraints, x0) || error("x0 = $x0 doesn't satisfy all constraints.")
    to, from = standardtransformation(lowerbound, upperbound)
    finternal = x -> f(to(x))
    cinternal = [x -> c(to(x)) for c in constraints]
    incumbent = from(x0)
    fincumbent = finternal(incumbent)
    for k in 1:max_iterations
        incumbent, fincumbent, i = search(m, finternal, cinternal, incumbent, fincumbent)
        if i != 1
            incumbent, fincumbent, i = poll(m, finternal, cinternal, incumbent, fincumbent)
        end
        update!(m.mesh, i)
        Δ(m.mesh) < min_mesh_size && return (f = fincumbent,
                                             x = to(incumbent),
                                             stopping_reason = MinMeshSize,
                                             iterations = k)
        if verbosity > Silent
            @show k to(incumbent) fincumbent i Δ(m.mesh)
        end
    end
    (f = fincumbent, x = to(incumbent), stopping_reason = MaxIterations,
     iterations = max_iteration)
end
end # module

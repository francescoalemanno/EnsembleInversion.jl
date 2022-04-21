"""
    EnsembleInversion.jl
exports `EKI` function, for further help type `?EKI` in the REPL.
"""
module EnsembleInversion
using LinearAlgebra, Statistics
"""
    EKI(G::Function, x_ensemble::Matrix{T}, y_target=zero(T), alpha=T(1e-8)*I)
solve model `y = G(x)` according to:

    `G` : model that takes a Vector `x` of dimension `D` and gives a predicted observation Vector `y` of size `K`

    `x_ensemble` : Matrix of size D-by-N, initial population of candidate solutions

    `y_target` : Vector of size K, representing the target model observation

    `alpha` : Covariance matrix of size K-by-K acting as a regularization

returns a new `x_ensemble` matrix with lower prediction error.
"""
function EKI(G::F, x_ensemble::AbstractMatrix{T}, y_target=zero(T), alpha=T(1e-8)*I) where {T <: Number, F <: Function}
    y_observed = mapreduce(hcat,eachcol(x_ensemble)) do p
        G(p)
    end
    J = size(x_ensemble,2)-1
    mx_ensemble = mean(x_ensemble,dims=2)
    my_observed = mean(y_observed,dims=2)
    dO = y_observed.-my_observed
    dP = x_ensemble.-mx_ensemble
    A = ((dP*dO')./J)*inv((dO*dO')./J + alpha)
    x_ensemble .+ A*(y_target.-y_observed)
end

export EKI
end

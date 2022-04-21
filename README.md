# EnsembleInversion.jl
    EKI(G::Function, x_ensemble::Matrix{T}, y_target=zero(T), alpha=T(1e-8)*I)
solve w.r.t `x` model `y = G(x)` according to:

    `G` : model that takes a Vector `x` of dimension `D` and gives a predicted observation Vector `y` of size `K`

    `x_ensemble` : Matrix of size D-by-N, initial population of candidate solutions

    `y_target` : Vector of size K, representing the target model observation

    `alpha` : Covariance matrix of size K-by-K acting as a regularization

returns a new `x_ensemble` matrix with lower prediction error.
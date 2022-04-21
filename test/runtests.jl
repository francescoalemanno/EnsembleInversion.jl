using EnsembleInversion, Test, Random

@testset "Vector Model" begin
    rng = Random.MersenneTwister(1023)
    function brownianrms((μ, σ), N)
        t = 0:N
        @.(sqrt(μ * μ * t * t + σ * σ * t))
    end
    params = (0.5, 2.0)
    tdata = brownianrms(params, 30)
    rg = LinRange(0,1,40)
    bp = [shuffle(rng,rg) shuffle(rng,rg).*4]'
    for i in 1:50
        bp = EKI(bp) do x
            brownianrms(x, 30) .- tdata
        end
    end
    est = sum(bp,dims=2)./size(bp,2)
    @test abs(est[1] - params[1]) < 1e-4
    @test abs(est[2] - params[2]) < 1e-4
end

@testset "Augmented Cost" begin
    rng = Random.MersenneTwister(1023)
    X = LinRange(-3,3,20)
    @. predictor(X,pp) = sin((X*pp[1] + pp[2])^2)
    params=[-0.65,1.2]
    Y = predictor(X,params)
    P = randn(rng,2,500)./2 .+ [-0.5,1]
    for i = 1:10
        P .= EKI(P,0.0) do pp
            predictor(X,pp).-Y
        end 
    end
    mP = vec(sum(P,dims=2) ./ size(P, 2))
    @test sum(abs2,Y .- predictor(X,mP)) < 1e-10
    @show sum(abs2,mP.-params) < 1e-12
end
using DynamicalSystems
using FileIO

"""
    lorenz96multi(J::Int, K::Int, u0 = rand((1+J)*K); F=0.01)

```math
\\frac{dx_i}{dt} = (x_{i+1}-x_{i-2})x_{i-1} - x_i + F
```

`N` is the chain length, `F` the forcing. Jacobian is created automatically.
(parameter container only contains `F`)
"""

function lorenz96multi(J::Int, K::Int, u0 = rand(K+J*K); Fˣ=10.,Fʸ=10.,h=1.,c=10.,b=10.)
    @assert J ≥ 4 "`J` must be at least 4"
    @assert K ≥ 4 "`K` must be at least 4"
    lor96m = Lorenz96multi{J,K}() # create struct
    return DynamicalSystemsBase.CDS(lor96m, u0, [Fˣ,Fʸ,h,c,b])
end
struct Lorenz96multi{J,K} end # Structure for size type
function (obj::Lorenz96multi{J,K})(dx, x, p, t) where {J,K}
    Fˣ = p[1]
    Fʸ = p[2]
    h = p[3]
    c = p[4]
    b = p[5]
    
    # convective scales (y)
    dy = reshape(view(dx,(K+1):(K+J*K)),K,J)
    y = reshape(view(x,(K+1):(K+J*K)),K,J)
    
    # large scales (x)
    # 3 large scale edge cases
    @inbounds dx[1] = (x[2] - x[K - 1]) * x[K]     - x[1] + Fˣ - (h*c/b)*sum(y[1,:])
    @inbounds dx[2] = (x[3] - x[K])     * x[1]     - x[2] + Fˣ - (h*c/b)*sum(y[2,:])
    @inbounds dx[K] = (x[1] - x[K - 2]) * x[K - 1] - x[K] + Fˣ - (h*c/b)*sum(y[K,:])

    # then the large scale general case
    for k in 3:(K - 1)
      @inbounds dx[k] = (x[k + 1] - x[k - 2]) * x[k - 1] - x[k] + Fˣ - (h*c/b)*sum(y[k,:])
    end
    
    # convective edge cases
    # k = 1
    @inbounds dy[1,1] = c*b*(y[1,2] - y[K,J-1]) * y[K,J] - c*y[1,1] + (c/b)*Fʸ + (h*c/b)*x[1]
    @inbounds dy[1,2] = c*b*(y[1,3] - y[K,J]) * y[1,1] - c*y[1,2] + (c/b)*Fʸ + (h*c/b)*x[1]
    @inbounds dy[1,J] = c*b*(y[2,1] - y[1,J-2]) * y[1,J-1] - c*y[1,J] + (c/b)*Fʸ + (h*c/b)*x[1]
    # k = K
    @inbounds dy[K,1] = c*b*(y[K,2] - y[K-1,J-1]) * y[K-1,J] - c*y[K,1] + (c/b)*Fʸ + (h*c/b)*x[K]
    @inbounds dy[K,2] = c*b*(y[K,3] - y[K-1,J]) * y[K,1] - c*y[K,2] + (c/b)*Fʸ + (h*c/b)*x[K]
    @inbounds dy[K,J] = c*b*(y[1,1] - y[K,J-2]) * y[K,J-1] - c*y[K,J] + (c/b)*Fʸ + (h*c/b)*x[K]
    
    for k in 2:(K-1)
        @inbounds dy[k,1] = c*b*(y[k,2] - y[k-1,J-1]) * y[k-1,J] - c*y[k,1] + (c/b)*Fʸ + (h*c/b)*x[k]
        @inbounds dy[k,2] = c*b*(y[k,3] - y[k-1,J]) * y[k,1] - c*y[k,2] + (c/b)*Fʸ + (h*c/b)*x[k]
        @inbounds dy[k,J] = c*b*(y[k+1,1] - y[k,J-2]) * y[k,J-1] - c*y[k,J] + (c/b)*Fʸ + (h*c/b)*x[k]
    end

    for k in 1:K
        # then the convective general case
        for j in 3:(J - 1)
            @inbounds dy[k,j] = c*b*(y[k,j+1] - y[k,j-2]) * y[k,j-1] - c*y[k,j] + (c/b)*Fʸ + (h*c/b)*x[k]
        end
    end

    return nothing
end

lor96 = lorenz96multi(36,10,Fˣ=10.,Fʸ=10.,h=1.,c=10.,b=√(10*10))
λλ = ChaosTools.lyapunovs(lor96, 100., dt = 0.01, Ttr = 50.)

save("lyap.jld2","lambda",λλ)

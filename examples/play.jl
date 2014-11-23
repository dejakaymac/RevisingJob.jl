using Distributions

K = 40
k = 10

#--------------------------------------------------------------
# Set up the global parameters
N = 20                                       # Training set size
true_w = [-0.3 0.5 -0.9]                     # The answer
σ² = 0.2^2            
# Setup the data -----------
generating_f = (w) -> x -> w[1] + w[2]*x + w[3]*x.^3  # The main thing
true_line    = generating_f(true_w)          # Bind the true params
noise        = ()->rand(Normal(0, σ²))       # Noise itself ... actually σ²(k)
noisy        = (f) -> x -> f(x) + noise()    # Return a noisy function
#Z = 

# Data load 2/4 weeks -------
# Historical
Z_D = 


# current obs
x = [1:k] -0.7

# Current forecast (f_Xᵢ,... marginals)
#f_X =
# f_X = noisy
f_X = Any[Normal(i,i) for  i in [1:K]]
Z_f = Float64[quantile(Normal(0,1), cdf(f_X[i], x[i])) for i=[1:k]] 


### SHOULD USE GENERATING FUNCTION FOR THIS STUFF?
                             


# playing --------------
a = rand(Normal(0,1),(N,N))
Σ = cov(a)
MvNormal(Σ)
PDMats.PDMat(Σ)
# playing --------------



# Likelihood -----------

#li = MvNormal(Float64[i < k ? 0 : Inf for i=[1:K]])
li = MvNormal(
              Float64[i < k ? Z_f[i] : 0.0 for i=[1:K] ], # mu_li
              #diagm(Float64[i < k ? 0 : Inf for i=[1:K]])
              diagm(Float64[i < k ? 0.01 : Inf for i=[1:K]]) # Σ_li
              )

# Prior ---------------

# full (FullCov) 
μ_full = [mean(Z_D[:,i]) for i=[1:size(Z_D)[2]]]  # or mean of fitted gaussian???
Σ_full = cov(Z_D)


# std (FullCovMu0) - Standardised
μ_std = nothing # zeros()
Σ_std = Σ_full

# markov
#= Note on p. 30 of revising report it says
P(z_t|Σ) = P(z_t_1 |Σ)P(z_t_2 |z_t_1 , Σ)P(z_t_3 |z_t_2 , Σ) . . . P(z_tn |z_tn−1 , Σ),
but it should be
P(z_t_1:n|Σ) = P(z_t_1 |Σ)P(z_t_2 |z_t_1 , Σ)P(z_t_3 |z_t_2 , Σ) . . . P(z_tn |z_tn−1 , Σ),
or
P(z_t|Σ) = P(z_t_1 |Σ) Sum_z_t_2 P(z_t_2 |z_t_1 , Σ) sum_P(z_t_3 |z_t_2 , Σ) . . . P(z_tn |z_tn−1 , Σ),
 ... is this the same thing as originally written?
however, later it says
=#

# markov2 (second order markov chain)


# PGM1 (markov)


# PGM2 (prev hour + prev day)


# PGM3 (prev 2,3 hours)




using Distributions
using Winston


flatten{T}(a::Array{T,1}) =
    any(map(x->isa(x,Array),a))? flatten(vcat(map(flatten,a)...)): a
flatten{T}(a::Array{T}) = reshape(a,prod(size(a)))


K = 37
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
repo = "/home/jade/metservice/revising/research/sandpit"
zfile = joinpath(repo,"zmat_tl060.txt")
#zfile = joinpath(repo,"zmat_tl365.txt")
Z = readdlm(zfile)
Z = reshape(Z,prod(size(Z)))
filter!((x)->typeof(x)==Float64 ? true: false,Z)
m = 37 # = index("")
n = Int(length(Z)/m)
Z = reshape(Z, (n,m))
Z = convert(Array{Float64,2},Z)
Z_D = Any[]
for row in [1:size(Z,1)]
    if -9999.0 in Z[row,:] != true
        push!(Z_D, Z[row,:])
    end
end
Z_D = flatten(Z_D)
m = 37 # = index("")
n = Int(length(Z_D)/m)
Z_D = reshape(Z_D, (n,m))

    
# plot histograms of each column
fp = FramedPlot()
for j in [1:size(Z_D,2)]
    add(fp, Histogram(hist(Z_D[:,j])...))
    display(fp)
end

# figure()
# tab = Table(4, 8)
# row = 1
# for j in [1:32]
#     fp = FramedPlot()
#     add(fp, Histogram(hist(Z_D[:,j],20)...))
#     col = (j-1)%8+1
#     tab[row,col] = fp
#     if col == 8
#         row +=1
#     end
#     display(tab)
# end



# plot contour
figure()
im = imagesc(Z_D)
display(im)


# current obs
x = [1:k] -0.7

# Current forecast (f_Xᵢ,... marginals)
#f_X =
# f_X = noisy
f_X = Any[Normal(i,i) for  i in [1:K]]
Z_f = Float64[quantile(Normal(0,1), cdf(f_X[i], x[i])) for i=[1:k]] 


### SHOULD USE GENERATING FUNCTION FOR THIS STUFF?
                             


# # playing --------------
# a = rand(Normal(0,1),(N,N))
# Σ = cov(a)
# MvNormal(Σ)
# PDMats.PDMat(Σ)
# # playing --------------



# Likelihood -----------

#li = MvNormal(Float64[i < k ? 0 : Inf for i=[1:K]])
μ_li = Float64[i < k ? Z_f[i] : 0.0 for i=[1:K] ] # mu_li
Σ_li = diagm(Float64[i < k ? 0.01 : Inf for i=[1:K]]) # Σ_li
#Σ_li = diagm(Float64[i < k ? 0.01 : 1e5 for i=[1:K]]) # Σ_li
ϕ_li = MvNormal(μ_li, Σ_li)

# Prior ---------------

# full (FullCov) 
μ_full_pr = Float64[mean(Z_D[:,i]) for i=[1:size(Z_D,2)]]  # or mean of fitted gaussian???
Σ_full_pr = cov(Z_D)
ϕ_full_pr = MvNormal(μ_full_pr, Σ_full_pr)
# This is fine Σ_li only on denom
μ_full_po =   (eye(K) - Σ_full_pr/(Σ_full_pr + Σ_li)) * μ_full_pr
            + (Σ_full_pr/(Σ_full_pr + Σ_li)) *  μ_li
Σ_full_po = (eye(K) - Σ_full_pr/(Σ_full_pr + Σ_li)) * Σ_full_pr
ϕ_full_po = MvNormal(μ_full_po, Σ_full_po)
# # This is bad Σ_li on numerator
# μ_full_po2 =  (Σ_li/(Σ_full_pr + Σ_li)) * μ_full_pr
#             + (Σ_full_pr/(Σ_full_pr + Σ_li)) *  μ_li
# Σ_full_po2 = Σ_full_pr * Σ_li / (Σ_full_pr + Σ_li)
# ϕ_full_po2 = MvNormal(μ_full_po2, Σ_full_po2)

posterior((ϕ_full_pr,Σ_li), MvNormal,x)
#posterior((ϕ_full_pr,))


#Now map in and out to original forecast






# # std (FullCovMu0) - Standardised
# μ_std_pr = zeros(K)
# Σ_std_pr = Σ_full
# ϕ_std_pr = MvNormal(Σ_full_pr)




# # markov
# #= Note on p. 30 of revising report it says
# P(z_t|Σ) = P(z_t_1 |Σ)P(z_t_2 |z_t_1 , Σ)P(z_t_3 |z_t_2 , Σ) . . . P(z_tn |z_tn−1 , Σ),
# but it should be
# P(z_t_1:n|Σ) = P(z_t_1 |Σ)P(z_t_2 |z_t_1 , Σ)P(z_t_3 |z_t_2 , Σ) . . . P(z_tn |z_tn−1 , Σ),
# or
# P(z_t|Σ) = P(z_t_1 |Σ) Sum_z_t_2 P(z_t_2 |z_t_1 , Σ) sum_P(z_t_3 |z_t_2 , Σ) . . . P(z_tn |z_tn−1 , Σ),
#  ... is this the same thing as originally written?
# however, later it says
# =#

# # markov2 (second order markov chain)


# # PGM1 (markov)


# # PGM2 (prev hour + prev day)


# # PGM3 (prev 2,3 hours)




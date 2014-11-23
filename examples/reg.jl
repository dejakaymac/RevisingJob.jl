# Pull in packages
using Distributions
using Winston
using GLM
using DataFrames
#--------------------------------------------------------------
# Set up the global parameters
N = 20                                       # Training set size
true_w = [-0.3 0.5 -0.9]                     # The answer
σ² = 0.2^2                                   # Variance of noise

# Nothing special here except to note that using unicode symbols seems
# to be idiomatic in Julia and is easily achieved in emacs using
# M-x set-input-method TeX

#--------------------------------------------------------------
#  Set up the data
# Line generating fns.  See how easy functional programming is, closures
# just fall right out of the syntax.
generating_f = (w) -> x -> w[1] + w[2]*x + w[3]*x.^3  # The main thing
true_line    = generating_f(true_w)          # Bind the true params
noise        = ()->rand(Normal(0, σ²))       # Noise itself
noisy        = (f) -> x -> f(x) + noise()    # Return a noisy function

# Sample random training x and y points.  Note that I have to match
# the columns with the generating function.  Not so awesome from a
# coding perspective.
function generate_data(f, n)
 x = -1 + 2*rand(n)
 y = map(f, x)
 ([ones(n,1) x x.^3], y)
end

# Thanks to Kevin for pointing out how to destructure this!
(X, y) = generate_data(noisy(true_line), N)

#--------------------------------------------------------------
#  Classic Linear Regression the MATLAB way.
#  This uses solves the linear system w'X = y using the QR decomp
w_matlab = X\y


#--------------------------------------------------------------
#  Classic Linear Regression the R way
#  This bottoms out to precisely the same QR decomp as the \ operator

#  This only seems contorted compared to the Matlab above because I
#  set the data up specifically for the Matlab operator and now have
#  to remunge it.
df = DataFrame(x  = X[:,2],
               x³ = X[:,3],
               y  = y)
model = fit(LinearModel, y ~ x + x³, df)
w_R = coef(model)


#--------------------------------------------------------------
#  Bayesian Linear Regression the Julia way !

# The prior is a 3D isotropic Normal. Use ₀ to indicate prior params.
μ₀ = [0.0, 0.0, 0.0]               # Prior assumes zero mean on w
Σ₀ = eye(3)*0.8                    # Prior is isotropic and wide
Σ₀ = eye(3)*0.1
prior = MvNormal(μ₀, Σ₀)

# Now update the prior with the data.  Use ₁ to indicate posterior params.
# I'm cheekily assuming I know the noise variance σ².
Σ₁ = inv(inv(Σ₀) + 1/σ²*X'*X)         # Equations (7.55) in MLAPP
μ₁ = Σ₁*inv(Σ₀)*μ₀ + 1/σ²*Σ₁*X'*y
posterior = MvNormal(μ₁, Σ₁)

# Sample from the posterior.  Beautiful.
w_samples = rand(posterior, 20*N)
#w_samples = rand(posterior, 200*N)

#--------------------------------------------------------------
# Plot our answers

domain = [-1.0 : 0.05 : 1.0]
p = FramedPlot()
setattr(p, "xrange", (-1,1))
setattr(p, "yrange", (-1,1))
setattr(p, "aspect_ratio", 1)
setattr(p, "title", "Regression Comparison")

# Plot samples from the posterior
so = Nothing
for i=1:(20*N)
  so = Curve(domain, generating_f(w_samples[:,i])(domain), "type", "dotted", "color", "grey" )
  add(p, so)
  setattr(so, "label", "Posterior Samples")
end

# Plot the posterior mean
w_postmean = mean(posterior)
s = Curve(domain, generating_f(w_postmean)(domain), "color", "blue" )
setattr(s, "label", "Posterior Mean")
add(p, s)

# Plot the MATLAB and R answers
c = Curve(domain, generating_f(w_matlab)(domain), "color", "orange")
setattr(c, "label", "Classic Linear Regression")
add(p, c)

# Plot the true answer
ct = Curve(domain, true_line(domain), "color", "red")
setattr(ct, "label", "True")
add(p, ct)

# Plot the data we used for inference
pt = Points( X[:,2], y, "type", "filled circle", "color", "red" )
setattr(pt, "label", "Training Data")
add(p, pt)

#l = Legend( .1, .9, {so, s, c, ct, pt} )
l = Legend( .1, .9, Any[so, s, c, ct, pt] )
add(p, l)

savefig(p, "posterior.png")
Winston.display(p)

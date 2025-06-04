using GaussianProcesses
using Random
using Plots  #Load Plots.jl package

Random.seed!(20140430)
# Training data

x = [0.0, 2.0, 2.3, 4.0, 2 * pi];               # predictors
y = sin.(x);                    # regressors

#Select mean and covariance function
mZero = MeanZero()                   # Zero mean function
kern = SE(0.0,0.0)                   # Sqaured exponential kernel (note that hyperparameters are on the log scale)


gp = GP(x,y,mZero,kern,-10000)       # Fit the GP
plot(gp; xlabel="x", ylabel="y", legend=false, fmt=:pdf) 
savefig("figure.pdf")

x = cat(x, [3.5, 5.0], dims=1)
y = sin.(x)
gp = GP(x,y,mZero,kern,-10000)       # Fit the GP
plot(gp; xlabel="x", ylabel="y", legend=false, fmt=:pdf)
savefig("figure2.pdf")
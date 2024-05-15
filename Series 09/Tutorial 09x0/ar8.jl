# load packages
using CSV, DataFrames, StatsPlots, Turing, Random

# Read CSV data into a DataFrame
df = CSV.read("EXPORT_2015_monthly_annual.csv", DataFrame)

# Convert 'FREE_ON_BOARD' column to Float64, removing commas from the strings first
df[!, :FREE_ON_BOARD] = parse.(Float64, replace.(df[!, :FREE_ON_BOARD], "," => ""))

# Extract 'FREE_ON_BOARD' values and 'MONTH' values
X_data = df[!, :FREE_ON_BOARD]
months = parse.(Int, df[!, :MONTH])  # Assuming 'MONTH' column is in string format "01", "02", etc.

using Flux: onehotbatch, onecold
# One-hot encode months
month_ohe = onehotbatch(months, 1:12)

# set parameters
time = length(X_data)
true_sigma = 0.1

# define model
@model function mymodel(time, X, months)
    # Priors
    phi_1 ~ Normal(0, 1)
    phi_2 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    
    # Coefficients for month effects
    beta ~ filldist(Normal(0, 1), 12)
    
    # Likelihood
    X[1] ~ Normal(0, sigma)
    X[2] ~ Normal(0, sigma)
    for t in 3:time
        month_effect = dot(beta, month_ohe[:, t])
        mu = phi_1 * X[t-1] + phi_2 * X[t-2] + month_effect
        X[t] ~ Normal(mu, sigma)
    end
end

# Create an instance of the model with the data
model = mymodel(time, X_data, months)

sampler = NUTS()
samples = 1_000
chain = sample(model, sampler, samples)

# visualize results
plot(chain)

# Prediction setup
time_fcst = 10  # Number of time steps to forecast
X_fcst = Matrix{Float64}(undef, time_fcst+2, samples)  # Initialize matrix for forecasts
# Initialize the first two values for the forecast using the last observed values
X_fcst[1, :] .= X_data[time-1]
X_fcst[2, :] .= X_data[time]
# Generate forecasts for each sample
for col in 1:samples
    # Draw parameters from the posterior samples
    phi_1_fcst = rand(chain[:, 1, 1])
    phi_2_fcst = rand(chain[:, 2, 1])
    sigma_fcst = rand(chain[:, 3, 1])
    beta_fcst = rand(chain[:, 4:end, 1])
    
    for row in 3:(time_fcst+2)
        month_idx = mod(months[time] + (row - 2) - 1, 12) + 1
        month_effect = dot(beta_fcst, onehotbatch(month_idx, 1:12))
        mu = phi_1_fcst * X_fcst[row-1, col] + phi_2_fcst * X_fcst[row-2, col] + month_effect
        noise_fcst = rand(Normal(0, sigma_fcst))
        X_fcst[row, col] = mu + noise_fcst
    end
end

# Visualize predictions
ts_fcst = time:(time + time_fcst)  # Time steps for forecast
for i in 1:samples
    plot!(ts_fcst, X_fcst[2:end, i], 
        legend = false, 
        linewidth = 1, color = :green, alpha = 0.1
    )
end

# Visualize the mean of the predictions
X_fcst_mean = [
    mean(X_fcst[i, 1:samples]) for i in 2:(time_fcst+2)
]
plot!(ts_fcst, X_fcst_mean, 
    legend = false, 
    linewidth = 2, 
    color = :red, 
    linestyle = :dot
)
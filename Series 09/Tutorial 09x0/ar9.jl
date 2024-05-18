# Load packages
using CSV, DataFrames, StatsPlots, Turing, Random

# Read CSV data
df = CSV.read("EXPORT_2015_monthly_annual.csv", DataFrame)

# Convert 'FREE_ON_BOARD' column to Float64
df[!, :FREE_ON_BOARD] = parse.(Float64, replace.(df[!, :FREE_ON_BOARD], "," => ""))

# Extract 'MONTH' and 'FREE_ON_BOARD' columns
df_monthly = combine(groupby(df, :MONTH), :FREE_ON_BOARD => sum => :FREE_ON_BOARD_SUM)

# Extract 'FREE_ON_BOARD_SUM' values
X_data = df_monthly[!, :FREE_ON_BOARD_SUM]

# Set parameters
time = length(X_data)
true_sigma = 0.1

# Define model
@model function mymodel(time, X)
    # prior
    phi_1 ~ Normal(0, 1)
    phi_2 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    # likelihood
    X[1] ~ Normal(0, sigma)
    X[2] ~ Normal(0, sigma)
    for t in 3:time
        mu = phi_1 * X[t-1] + phi_2 * X[t-2]
        X[t] ~ Normal(mu, sigma)
    end
end

# Infer posterior probability
model = mymodel(time, X_data)
sampler = NUTS()
samples = 1_000
chain = sample(model, sampler, samples)

# Visualize results
plot(chain)

# Make predictions
time_fcst = 10
X_fcst = Matrix{Float64}(undef, time_fcst+2, samples)
X_fcst[1, :] .= X_data[time-1]
X_fcst[2, :] .= X_data[time]

for col in 1:samples
    phi_1_fcst = rand(chain[:, 1, 1])
    phi_2_fcst = rand(chain[:, 2, 1])
    error_fcst = rand(chain[:, 3, 1])
    noise_fcst = rand(Normal(0, error_fcst))
    for row in 3:(time_fcst+2)
        X_fcst[row, col] =
            phi_1_fcst * X_fcst[row-1, col] +
            phi_2_fcst * X_fcst[row-2, col] +
        noise_fcst
    end
end

# Visualize predictions
ts_fcst = time:(time + time_fcst)
for i in 1:samples
    plot!(ts_fcst, X_fcst[2:end, i],
        legend = false,
        linewidth = 1, color = :green, alpha = 0.1
    )
end

# Visualize mean values for predictions
X_fcst_mean = [
    mean(X_fcst[i, 1:samples]) for i in 2:(time_fcst+2)
]
plot!(ts_fcst, X_fcst_mean,
    legend = false,
    linewidth = 2,
    color = :red,
    linestyle = :dot
)

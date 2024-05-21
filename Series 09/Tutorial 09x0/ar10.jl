using CSV, DataFrames, StatsPlots, Turing, Random

# Read CSV data
df = CSV.read("EXPORT_2015_monthly_annual.csv", DataFrame)

# Convert 'FREE_ON_BOARD' column to Float64
df[!, :FREE_ON_BOARD] = parse.(Float64, replace.(df[!, :FREE_ON_BOARD], "," => ""))

# Extract 'MONTH' and 'FREE_ON_BOARD' columns
df_monthly = combine(groupby(df, :MONTH), :FREE_ON_BOARD => sum => :FREE_ON_BOARD_SUM)

# Print the summarized data for verification
println("Monthly summarized data:")
println(df_monthly)

# Extract 'FREE_ON_BOARD_SUM' values
X_data = df_monthly[!, :FREE_ON_BOARD_SUM]

# Print the data to be used in the model
println("X_data:")
println(X_data)

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

# Print the chain for verification
println("Chain:")
println(chain)

# Visualize results
plot(chain)

# Make predictions
time_fcst = 10
X_fcst = Matrix{Float64}(undef, time_fcst+2, samples)
X_fcst[1, :] .= X_data[time-1]
X_fcst[2, :] .= X_data[time]

for col in 1:samples
    phi_1_fcst = rand(chain[:phi_1])
    phi_2_fcst = rand(chain[:phi_2])
    sigma_fcst = rand(chain[:sigma])
    for row in 3:(time_fcst+2)
        X_fcst[row, col] = phi_1_fcst * X_fcst[row-1, col] + phi_2_fcst * X_fcst[row-2, col] + rand(Normal(0, sigma_fcst))
    end
end

# Visualize predictions
ts_fcst = time:(time + time_fcst)
plot(1:time, X_data, label="Actual Data", linewidth=2)
for i in 1:samples
    plot!(ts_fcst, X_fcst[2:end, i], legend=false, linewidth=1, color=:green, alpha=0.1)
end

# Visualize mean values for predictions
X_fcst_mean = [mean(X_fcst[i, :]) for i in 2:(time_fcst+2)]
plot!(ts_fcst, X_fcst_mean, legend=false, linewidth=2, color=:red, linestyle=:dot)

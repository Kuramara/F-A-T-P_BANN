using CSV, DataFrames, StatsPlots, Turing, Random

# Read CSV data
df_2015 = CSV.read("EXPORT_2015_monthly_annual.csv", DataFrame)
df_2016 = CSV.read("EXPORT_2016_monthly_annual.csv", DataFrame)

# Convert 'FREE_ON_BOARD' column to Float64
df_2015[!, :FREE_ON_BOARD] = parse.(Float64, replace.(df_2015[!, :FREE_ON_BOARD], "," => ""))
df_2016[!, :FREE_ON_BOARD] = parse.(Float64, replace.(df_2016[!, :FREE_ON_BOARD], "," => ""))

# Extract 'MONTH' and 'FREE_ON_BOARD' columns
df_monthly_2015 = combine(groupby(df_2015, :MONTH), :FREE_ON_BOARD => sum => :FREE_ON_BOARD_SUM)
df_monthly_2016 = combine(groupby(df_2016, :MONTH), :FREE_ON_BOARD => sum => :FREE_ON_BOARD_SUM)

# Print the summarized data for verification
println("Monthly summarized data for 2015:")
println(df_monthly_2015)
println("Monthly summarized data for 2016:")
println(df_monthly_2016)

# Extract 'FREE_ON_BOARD_SUM' values
X_data_2015 = df_monthly_2015[!, :FREE_ON_BOARD_SUM]
X_data_2016 = df_monthly_2016[!, :FREE_ON_BOARD_SUM]

# Print the data to be used in the model
println("X_data_2015:")
println(X_data_2015)
println("X_data_2016:")
println(X_data_2016)

# Set parameters
time = length(X_data_2015)
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
model = mymodel(time, X_data_2015)
sampler = NUTS()
samples = 1_000
chain = sample(model, sampler, samples)

# Print the chain for verification
println("Chain:")
println(chain)

# Visualize the posterior distribution
plot(chain, label=["phi_1" "phi_2" "sigma"], title="Posterior Distributions")

# Make predictions for 2016 based on 2015 data
time_fcst = 12  # Forecasting the next 12 months
X_fcst = Matrix{Float64}(undef, time_fcst+2, samples)
X_fcst[1, :] .= X_data_2015[end-1]
X_fcst[2, :] .= X_data_2015[end]

for col in 1:samples
    phi_1_fcst = rand(chain[:phi_1])
    phi_2_fcst = rand(chain[:phi_2])
    sigma_fcst = rand(chain[:sigma])
    for row in 3:(time_fcst+2)
        X_fcst[row, col] = phi_1_fcst * X_fcst[row-1, col] + phi_2_fcst * X_fcst[row-2, col] + rand(Normal(0, sigma_fcst))
    end
end

# Visualize the actual data for 2015
plot(1:time, X_data_2015, label="Actual Data 2015", linewidth=2, title="Actual Data 2015")

# Visualize the actual data for 2016
plot(1:12, X_data_2016, label="Actual Data 2016", linewidth=2, color=:blue, title="Actual Data 2016")

# Visualize the predictions vs actual data
ts_fcst = (time+1):(time + time_fcst)
p = plot(1:time, X_data_2015, label="Actual Data 2015", linewidth=2, title="Predictions vs Actual Data 2016")
plot!(p, ts_fcst, X_data_2016, label="Actual Data 2016", linewidth=2, color=:blue)

# Plot the individual forecast samples
for i in 1:samples
    plot!(p, ts_fcst, X_fcst[3:end, i], legend=false, linewidth=1, color=:green, alpha=0.1)
end

# Visualize mean values for predictions
X_fcst_mean = [mean(X_fcst[i, :]) for i in 3:(time_fcst+2)]
plot!(p, ts_fcst, X_fcst_mean, legend=false, linewidth=2, color=:red, linestyle=:dot, label="Predicted Mean")

# Show plot
display(p)

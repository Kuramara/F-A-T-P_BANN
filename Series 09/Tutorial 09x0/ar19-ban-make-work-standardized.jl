using CSV, DataFrames, StatsPlots, Turing, Random, Statistics, MCMCChains

# Function to preprocess data
function preprocess_data(filename)
    df = CSV.read(filename, DataFrame)
    df[!, :FREE_ON_BOARD] = coalesce.(df[!, :FREE_ON_BOARD], "0")  # Replace missing values with "0"
    df[!, :FREE_ON_BOARD] = parse.(Float64, replace.(df[!, :FREE_ON_BOARD], "," => ""))
    df_monthly = combine(groupby(df, :MONTH), :FREE_ON_BOARD => sum => :FREE_ON_BOARD_SUM)
    return df_monthly
end

# Read and preprocess CSV data
df_monthly_2015 = preprocess_data("EXPORT_2015_monthly_annual.csv")
df_monthly_2016 = preprocess_data("EXPORT_2016_monthly_annual.csv")
df_monthly_2017 = preprocess_data("EXPORT_2017_monthly_annual.csv")
df_monthly_2018 = preprocess_data("EXPORT_2018_monthly_annual.csv")
df_monthly_2019 = preprocess_data("EXPORT_2019_monthly_annual.csv")
df_monthly_2020 = preprocess_data("EXPORT_2020_monthly_annual.csv")
df_monthly_2021 = preprocess_data("EXPORT_2021_monthly_annual.csv")
df_monthly_2022 = preprocess_data("EXPORT_2022_monthly_annual.csv")
df_monthly_2023 = preprocess_data("EXPORT_2023_monthly_annual.csv")

# Combine 2015-2022 data
df_combined = vcat(df_monthly_2015, df_monthly_2016, df_monthly_2017, df_monthly_2018, df_monthly_2019, df_monthly_2020, df_monthly_2021, df_monthly_2022)

# Extract 'FREE_ON_BOARD_SUM' values
X_data_combined = df_combined[!, :FREE_ON_BOARD_SUM]
X_data_2023 = df_monthly_2023[!, :FREE_ON_BOARD_SUM]

# Set parameters
time = length(X_data_combined)
true_sigma = 0.1

# Define model
@model function mymodel(time, X)
    # Prior
    phi_1 ~ Normal(0, 1)
    phi_2 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    # Likelihood
    X[1] ~ Normal(0, sigma)
    X[2] ~ Normal(0, sigma)
    for t in 3:time
        mu = phi_1 * X[t-1] + phi_2 * X[t-2]
        X[t] ~ Normal(mu, sigma)
    end
end

# Infer posterior probability
model = mymodel(time, X_data_combined)
sampler = NUTS()
samples = 1000
chain = sample(model, sampler, samples)

# Make predictions for 2023 based on 2015-2022 data
time_fcst = 12  # Forecasting the next 12 months
X_fcst = Matrix{Float64}(undef, time_fcst+2, samples)
X_fcst[1, :] .= X_data_combined[end-1]
X_fcst[2, :] .= X_data_combined[end]

for col in 1:samples
    phi_1_fcst = rand(chain[:phi_1])
    phi_2_fcst = rand(chain[:phi_2])
    sigma_fcst = rand(chain[:sigma])
    for row in 3:(time_fcst+2)
        X_fcst[row, col] = phi_1_fcst * X_fcst[row-1, col] + phi_2_fcst * X_fcst[row-2, col] + rand(Normal(0, sigma_fcst))
    end
end

# Calculate mean forecast values
X_fcst_mean = [mean(X_fcst[i, :]) for i in 3:(time_fcst+2)]

# Calculate MAPE and MSE
mape = mean(abs.((X_data_2023 - X_fcst_mean) ./ X_data_2023)) * 100
mse = mean((X_data_2023 - X_fcst_mean).^2)

# Calculate 95% confidence intervals
lower_bound = [quantile(X_fcst[i, :], 0.025) for i in 3:(time_fcst+2)]
upper_bound = [quantile(X_fcst[i, :], 0.975) for i in 3:(time_fcst+2)]

# Print MAPE and MSE
println("MAPE: $mape%")
println("MSE: $mse")

# Visualize the predictions vs actual data
ts_fcst = (time+1):(time + time_fcst)
p = plot(1:time, X_data_combined, label="Actual Data 2015-2022", linewidth=2, title="Predictions vs Actual Data 2023", xlabel="Time", ylabel="FREE_ON_BOARD_SUM")
plot!(p, ts_fcst, X_data_2023, label="Actual Data 2023", linewidth=2, color=:blue)

# Plot the individual forecast samples
for i in 1:samples
    plot!(p, ts_fcst, X_fcst[3:end, i], legend=false, linewidth=1, color=:green, alpha=0.1)
end

# Visualize mean values for predictions and confidence intervals
plot!(p, ts_fcst, X_fcst_mean, legend=false, linewidth=2, color=:red, linestyle=:dot, label="Predicted Mean")
plot!(p, ts_fcst, lower_bound, legend=false, linewidth=1, color=:black, linestyle=:dash, label="95% CI Lower Bound")
plot!(p, ts_fcst, upper_bound, legend=false, linewidth=1, color=:black, linestyle=:dash, label="95% CI Upper Bound")

# Show plot
display(p)

# Check convergence diagnostics
summary(chain)
plot(chain)

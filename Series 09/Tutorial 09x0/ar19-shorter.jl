using CSV, DataFrames, StatsPlots, Turing, Random, LinearAlgebra, Statistics

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
# Comment out other years
# df_monthly_2018 = preprocess_data("EXPORT_2018_monthly_annual.csv")
# df_monthly_2019 = preprocess_data("EXPORT_2019_monthly_annual.csv")
# df_monthly_2020 = preprocess_data("EXPORT_2020_monthly_annual.csv")
# df_monthly_2021 = preprocess_data("EXPORT_2021_monthly_annual.csv")
# df_monthly_2022 = preprocess_data("EXPORT_2022_monthly_annual.csv")
# df_monthly_2023 = preprocess_data("EXPORT_2023_monthly_annual.csv")

# Combine 2015-2016 data
df_combined = vcat(df_monthly_2015, df_monthly_2016)

# Extract 'FREE_ON_BOARD_SUM' values for training and prediction
X_data_combined = df_combined[!, :FREE_ON_BOARD_SUM]
X_data_2017 = df_monthly_2017[!, :FREE_ON_BOARD_SUM]

# Normalize the data
X_data_combined_norm = (X_data_combined .- mean(X_data_combined)) ./ std(X_data_combined)
X_data_2017_norm = (X_data_2017 .- mean(X_data_2017)) ./ std(X_data_2017)

# Set parameters
time = length(X_data_combined_norm)

# Define Bayesian Neural Network model
@model function BNN_1(X::Matrix{Float32}, y::Vector{Float32}, nn::Chain, ::Type{T} = Float32) where {T}
	# priors
	α ~ truncated(Normal(0, 5); lower=0.0001)
	β ~ truncated(Normal(0, 5); lower=0.0001)
	σ ~ InverseGamma(α, β)

	μ = (nn(X')')[:]

	# Likelihood
	y ~ MvNormal(μ, σ * I)
	return Nothing
end

# Prepare data for BNN
X = collect(1:time)'
y = X_data_combined_norm

# Infer posterior probability
model = bnn(X, y)
sampler = NUTS()
samples = 1_000
chain = sample(model, sampler, samples)

# Debugging: Check the available parameters in the chain
println("Chain parameter names: ", names(chain))

# Make predictions for 2017 based on 2015-2016 data
time_fcst = 12  # Forecasting the next 12 months
X_fcst = Matrix{Float64}(undef, time_fcst, samples)

for col in 1:samples
    w1_fcst = chain[:w1][col, :]
    b1_fcst = chain[:b1][col, :]
    w2_fcst = chain[:w2][col, :]
    b2_fcst = chain[:b2][col, :]
    
    for row in 1:time_fcst
        h = tanh.(w1_fcst .* (time + row) .+ b1_fcst)
        X_fcst[row, col] = sum(w2_fcst .* h) + sum(b2_fcst)
    end
end

# Denormalize the predictions
X_fcst_denorm = X_fcst .* std(X_data_combined) .+ mean(X_data_combined)

# Calculate mean forecast values
X_fcst_mean = [mean(X_fcst_denorm[i, :]) for i in 1:time_fcst]

# Calculate MAPE and MSE
mape = mean(abs.((X_data_2017 - X_fcst_mean) ./ X_data_2017)) * 100
mse = mean((X_data_2017 - X_fcst_mean).^2)

# Calculate 95% confidence intervals
lower_bound = [quantile(X_fcst_denorm[i, :], 0.025) for i in 1:time_fcst]
upper_bound = [quantile(X_fcst_denorm[i, :], 0.975) for i in 1:time_fcst]

# Print MAPE and MSE
println("MAPE: $mape%")
println("MSE: $mse")

# Visualize the predictions vs actual data
ts_fcst = (time+1):(time + time_fcst)
p = plot(1:time, X_data_combined, label="Actual Data 2015-2016", linewidth=2, title="Predictions vs Actual Data 2017")
plot!(p, ts_fcst, X_data_2017, label="Actual Data 2017", linewidth=2, color=:blue)

# Plot the individual forecast samples
for i in 1:samples
    plot!(p, ts_fcst, X_fcst_denorm[i, :], legend=false, linewidth=1, color=:green, alpha=0.1)
end

# Visualize mean values for predictions and confidence intervals
plot!(p, ts_fcst, X_fcst_mean, legend=false, linewidth=2, color=:red, linestyle=:dot, label="Predicted Mean")
plot!(p, ts_fcst, lower_bound, legend=false, linewidth=1, color=:black, linestyle=:dash, label="95% CI Lower Bound")
plot!(p, ts_fcst, upper_bound, legend=false, linewidth=1, color=:black, linestyle=:dash, label="95% CI Upper Bound")

# Show plot
display(p)

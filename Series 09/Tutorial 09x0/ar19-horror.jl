using CSV, DataFrames, StatsPlots, Turing, Random, Statistics
using Printf

# Function to preprocess data
function preprocess_data(filename)
    df = CSV.read(filename, DataFrame)
    df[!, :FREE_ON_BOARD] = coalesce.(df[!, :FREE_ON_BOARD], "0")  # Replace missing values with "0"
    df[!, :FREE_ON_BOARD] = parse.(Float64, replace.(df[!, :FREE_ON_BOARD], "," => ""))
    df_monthly = combine(groupby(df, [:COMMODITY_DESCRIPTION, :MONTH]), :FREE_ON_BOARD => sum => :FREE_ON_BOARD_SUM)
    return df_monthly
end

# Read and preprocess CSV data for each year
data_files = [
    "EXPORT_2015_monthly_annual.csv",
    "EXPORT_2016_monthly_annual.csv",
    "EXPORT_2017_monthly_annual.csv"
]

dfs = [preprocess_data(file) for file in data_files]
df_combined = vcat(dfs[1:end-1]...)  # Combine 2015-2017 data

# Filter data for the last 12 months
last_12_months_df = filter(row -> (row.MONTH >= 1 && row.MONTH <= 12), df_combined)

# Sum up the FOB values for each commodity
df_total_fob = combine(groupby(last_12_months_df, :COMMODITY_DESCRIPTION), :FREE_ON_BOARD_SUM => sum => :TOTAL_FOB)

# Sort and select top 10 commodities
top_10_commodities = first(sort(df_total_fob, :TOTAL_FOB, rev=true), 10).COMMODITY_DESCRIPTION

# Filter data for top 10 commodities
df_top_10_combined = filter(row -> row[:COMMODITY_DESCRIPTION] in top_10_commodities, last_12_months_df)

# Calculate and print sum of FOB for each commodity
for commodity in top_10_commodities
    sum_fob = sum(df_top_10_combined[df_top_10_combined[:, :COMMODITY_DESCRIPTION] .== commodity, :FREE_ON_BOARD_SUM])
    @printf("Sum of FOB for %s: %0.2f\n", commodity, sum_fob)
end

# Define forecasting model
@model function mymodel(time, X)
    # priors
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

function forecast_commodity(df_combined, commodity)
    # Prepare data for the specific commodity
    df_commodity_combined = filter(row -> row[:COMMODITY_DESCRIPTION] == commodity, df_combined)
    
    X_data_combined = df_commodity_combined[!, :FREE_ON_BOARD_SUM]
    
    time = length(X_data_combined)
    
    # Run Turing model
    model = mymodel(time, X_data_combined)
    sampler = NUTS()
    samples = 1_000
    chain = sample(model, sampler, samples)
    
    # Forecasting the next 12 months
    time_fcst = 12
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
    X_fcst_mean = [mean(X_fcst[row, :]) for row in 3:(time_fcst+2)]  # Calculate mean for each row, not each column
    
    # Calculate MAE, RMSE, and MAPE
    mae = mean(abs.(X_data_combined[end-11:end] - X_fcst_mean[1:12]))
    rmse = sqrt(mean((X_data_combined[end-11:end] - X_fcst_mean[1:12]).^2))
    mape = mean(abs.((X_data_combined[end-11:end] - X_fcst_mean[1:12]) ./ X_data_combined[end-11:end])) * 100
    
    println("MAE for $commodity: $mae")
    println("RMSE for $commodity: $rmse")
    println("MAPE for $commodity: $mape%")

    # Calculate and print sum of FOB for the last 12 months of the predicted values
    sum_fob_fcst_last_12_months = sum(X_fcst[end-11:end, :], dims=1)
    total_sum_fob_fcst_last_12_months = sum(sum_fob_fcst_last_12_months)
    @printf("Sum of FOB for the last 12 months of the predicted values for %s: %0.2f\n", commodity, total_sum_fob_fcst_last_12_months)

    # Visualize the predictions vs actual data
    ts_fcst = (time+1):(time + time_fcst)
    p = plot(1:(time+time_fcst), vcat(X_data_combined, X_fcst_mean), label="Actual Data", linewidth=2, title="Predictions vs Actual Data for $commodity")
    
    # Plot the individual forecast samples
    for i in 1:samples
        plot!(p, ts_fcst, X_fcst[3:end, i], legend=false, linewidth=1, color=:green, alpha=0.1)
    end
    
    # Visualize mean values for predictions
    plot!(p, ts_fcst, X_fcst_mean, legend=false, linewidth=2, color=:red, linestyle=:dot, label="Predicted Mean")
    
    # Show plot
    display(p)
end

# Forecast for each of the top 10 commodities
for commodity in top_10_com

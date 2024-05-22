using CSV, DataFrames, StatsPlots, Turing, Random, Statistics

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
    "EXPORT_2016_monthly_annual.csv"
    # "EXPORT_2017_monthly_annual.csv"
    # "EXPORT_2018_monthly_annual.csv",
    # "EXPORT_2019_monthly_annual.csv",
    # "EXPORT_2020_monthly_annual.csv",
    # "EXPORT_2021_monthly_annual.csv",
    # "EXPORT_2022_monthly_annual.csv",
    # "EXPORT_2023_monthly_annual.csv"
]

dfs = [preprocess_data(file) for file in data_files]
df_combined = vcat(dfs[1:end-1]...)  # Combine 2015-2022 data

# Sum up the FOB values for each commodity
df_total_fob = combine(groupby(df_combined, :COMMODITY_DESCRIPTION), :FREE_ON_BOARD_SUM => sum => :TOTAL_FOB)

# Sort and select top 10 commodities
top_10_commodities = first(sort(df_total_fob, :TOTAL_FOB, rev=true), 10).COMMODITY_DESCRIPTION

# Filter data for top 10 commodities
df_top_10_combined = filter(row -> row[:COMMODITY_DESCRIPTION] in top_10_commodities, df_combined)

# Calculate and print sum of FOB for each commodity
using Printf

# Get the first 12 months of data
first_12_months = df_top_10_combined[1:12, :]

# Calculate and print sum of FOB for each commodity
for commodity in top_10_commodities
    sum_fob = sum(first_12_months[first_12_months[:, :COMMODITY_DESCRIPTION] .== commodity, :FREE_ON_BOARD_SUM])
    @printf("Sum of FOB for %s in the first 12 months: %0.2f\n", commodity, sum_fob)
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
    # Print out the top 10 predicted sums
    X_fcst_sorted = sort(X_fcst_mean, rev=true)
    println("Month predicted sums:")
    for i in 1:min(12, length(X_fcst_sorted))
        println("Prediction $i: $(X_fcst_sorted[i])")
    end
    overall_sum = sum(X_fcst_mean)
    println("Overall sum of predicted values: $(@sprintf("%.2f", overall_sum))")
    # println("MAE for $commodity: $(@sprintf("%.2f", mae))")
    # println("RMSE for $commodity: $(@sprintf("%.2f", rmse))")
    # println("MAPE for $commodity: $(@sprintf("%.2f", mape))%")
    println("MAE for $commodity: $mae")
    println("RMSE for $commodity: $rmse")
    println("MAPE for $commodity: $mape%")
    
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
for commodity in top_10_commodities
    forecast_commodity(df_top_10_combined, commodity)
end

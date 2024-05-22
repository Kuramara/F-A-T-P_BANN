using CSV
using DataFrames

# Read the CSV data
data = CSV.read("EXPORT_2015_monthly_annual.csv", DataFrame)

# Convert FOB to numeric
data.FREE_ON_BOARD = parse.(Float64, replace.(data.FREE_ON_BOARD, "," => ""))

# Group by month and calculate the sum of FOB
monthly_fob = combine(groupby(data, :MONTH), :FREE_ON_BOARD => sum)

# Find the top 10 commodities by FOB
top_commodities = sort(combine(groupby(data, :COMMODITY_DESCRIPTION), :FREE_ON_BOARD => sum), :FREE_ON_BOARD_sum, rev=true)[1:10, :]

println(monthly_fob)
println(top_commodities)

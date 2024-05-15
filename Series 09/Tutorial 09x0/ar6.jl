using CSV, DataFrames, Dates, TimeSeries, Flux, Turing, Distributions, MCMCChains

# Load the data
df = CSV.read("EXPORT_2015_monthly_annual.csv", DataFrame)

# Convert the FOB column to numeric (removing commas)
df[!,:FREE_ON_BOARD] = parse.(Float64, replace.(df.FREE_ON_BOARD, "," => ""))
# Create a time series where each observation is associated with a specific month
dates = Date.(string.(df.MONTH) .* "-01")  # Assuming df.MONTH contains the month number
ts = TimeArray(dates, df.FREE_ON_BOARD)

# Calculate the sum of FOB values for each month
monthly_sums = combine(groupby(df, :MONTH), :FREE_ON_BOARD => sum)

# Prepare the data for training
x = Matrix(monthly_sums[1:end-1, 2])  # Input: current month's sum of FOB values
y = Matrix(monthly_sums[2:end, 2])    # Output: next month's sum of FOB values

# Define the Bayesian neural network model
function bayesian_nn(input_size, hidden_size, output_size)
    return Chain(
        Dense(input_size, hidden_size, σ),
        Dense(hidden_size, output_size)
    )
end

# Specify the Bayesian model
@model function bnn_model(x, y, input_size, hidden_size, output_size)
    # Priors
    W1 ~ filldist(Normal(0, 1), hidden_size, input_size)
    b1 ~ filldist(Normal(0, 1), hidden_size)
    W2 ~ filldist(Normal(0, 1), output_size, hidden_size)
    b2 ~ filldist(Normal(0, 1), output_size)

    # Neural network
    nn = bayesian_nn(input_size, hidden_size, output_size)
    
    # Likelihood
    y_pred = nn(x * W1 .+ b1)
    y_pred = y_pred * W2 .+ b2
    y ~ MvNormal(vec(y_pred), 1)
end

# Define the input, hidden, and output sizes for the neural network
input_size = 1
hidden_size = 10
output_size = 1

# Define the number of samples for the Bayesian inference
num_samples = 1000

# Perform Bayesian inference
chain = sample(bnn_model(x, y, input_size, hidden_size, output_size), NUTS(), num_samples)

# Print a summary of the chain
println(summarystats(chain))

# Extract the samples
posterior_samples = MCMCChains.group(chain, :W1, :b1, :W2, :b2)

# Predict next month's sum of FOB values
function predict(model, x_new, posterior_samples)
    # Use the posterior samples to generate predictions
    predictions = [model(x_new, posterior_samples[1][:,i]) for i in 1:size(posterior_samples[1], 2)]
    y_pred = mean(predictions)
    return y_pred
end

# Prepare input for prediction
x_new = [monthly_sums[end, 2]]  # Using the last observed sum of FOB values to predict the next month's sum

# Make the prediction
prediction = predict(bayesian_nn, x_new, posterior_samples)

println("Predicted next month's sum of FOB values: $prediction")

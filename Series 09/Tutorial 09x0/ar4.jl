using CSV, DataFrames, Dates, TimeSeries, Flux, Turing, Distributions

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
x = values(monthly_sums[1:end-1, 2])  # Input: current month's sum of FOB values
y = values(monthly_sums[2:end, 2])    # Output: next month's sum of FOB values

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
    W1 ~ Normal(0, 1)
    b1 ~ Normal(0, 1)
    W2 ~ Normal(0, 1)
    b2 ~ Normal(0, 1)
    
    # Convert input to arrays
    x_array = reshape(Vector{Float32}(float.(x)), 1, :)
    y_array = reshape(Vector{Float32}(float.(y)), 1, :)
    # Bayesian neural network
    nn = bayesian_nn(input_size, hidden_size, output_size)
    
    # Likelihood
    y_pred = vec(nn(x_array))
    y_array ~ MvNormal(y_pred, 1)

    # y_pred = nn(x_array)
    # y ~ MvNormal(y_pred, 1)
end

# Define the input, hidden, and output sizes for the neural network
input_size = size(x, 2)
hidden_size = 10
output_size = size(y, 2)

# Define the number of samples for the Bayesian inference
num_samples = 1000

# Perform Bayesian inference
chain = sample(bnn_model(x, y, input_size, hidden_size, output_size), NUTS(), num_samples)

# Print a summary of the chain
println(summarystats(chain))

# Get the posterior distribution
posterior = get(chain, :parameters)

# Predict next month's sum of FOB values
function predict(model, x_new, posterior)
    # Use the posterior distribution to generate predictions
    y_pred = mean([model(rand(posterior[:W1]), rand(posterior[:b1]), rand(posterior[:W2]), rand(posterior[:b2]))(x_new) for _ in 1:1000])

    return y_pred
end


# Prepare input for prediction
x_new = monthly_sums[end, 2]  # Using the last observed sum of FOB values to predict the next month's sum

# Make the prediction
prediction = predict(bayesian_nn, x_new, posterior)

println("Predicted next month's sum of FOB values: $prediction")
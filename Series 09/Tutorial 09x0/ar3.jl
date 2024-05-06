using CSV, DataFrames, Dates, TimeSeries, Flux, Turing, Distributions

# Load the data
df = CSV.read("EXPORT_2015_monthly_annual.csv", DataFrame)

# Convert the FOB column to numeric (removing commas)
df[!,:FREE_ON_BOARD] = parse.(Float64, replace.(df.FREE_ON_BOARD, "," => ""))
# Create a time series where each observation is associated with a specific month
dates = Date.(string.(df.MONTH) .* "-01")  # Assuming df.MONTH contains the month number
ts = TimeArray(dates, df.FREE_ON_BOARD)

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
    x_array = Matrix{Float32}(float.(Matrix(x')))
    y_array = Matrix{Float32}(float.(Matrix(y')))
    
    # Bayesian neural network
    nn = Chain(
        Dense(input_size, hidden_size, σ),
        Dense(hidden_size, output_size)
    )
    
    # Likelihood
    y_pred = nn(x_array)
    y_pred = y_pred
    y_array = y_array
    y_pred = y_pred[1,:]
    y_array = y_array[1,:]
    y ~ MvNormal(y_pred, 1)
end




# Prepare the data for training
x = reshape(values(ts[1:end-1]), :, 1)  # Convert Vector to Matrix
y = reshape(values(ts[2:end]), :, 1)    # Convert Vector to Matrix


# Define the input, hidden, and output sizes for the neural network
input_size = size(x, 2)
hidden_size = 10
output_size = size(y, 2)

# Define the number of samples for the Bayesian inference
num_samples = 1000

# Perform Bayesian inference
chain = sample(bnn_model(x, y, input_size, hidden_size, output_size), NUTS(), num_samples)


# Get the posterior distribution
posterior = get(chain, :parameters)
if isempty(posterior)
    println("The posterior is empty.")
else
    println("The posterior is not empty.")
end

W1_samples = chain[:W1]
b1_samples = chain[:b1]
W2_samples = chain[:W2]
b2_samples = chain[:b2]

if isempty(W1_samples) || isempty(b1_samples) || isempty(W2_samples) || isempty(b2_samples)
    println("The posterior is empty.")
else
    println("The posterior is not empty.")
end

#Predicting
function predict(model, x_new, posterior)
    # Convert x_new to the correct format
    x_new = reshape(values(x_new), :, 1)  # Convert TimeArray to Matrix

    # Use the posterior distribution to generate predictions
    y_pred = mean([model(rand(posterior.W1), rand(posterior.b1), rand(posterior.W2), rand(posterior.b2))(x_new) for _ in 1:1000])

    return y_pred
end


posterior = (
    W1 = chain[:W1],
    b1 = chain[:b1],
    W2 = chain[:W2],
    b2 = chain[:b2]
)


# Prepare input for prediction
x_new = values(ts[end])  # Using the last observed FOB value to predict the next month

# Make the prediction
prediction = predict(bayesian_nn, x_new, posterior)

println("Predicted next month's FOB value: $prediction")


# #Visualization
# using Plots

# # Create a histogram for each parameter
# p1 = histogram(chain[:W1], bins=50, title="Posterior of W1", label="")
# p2 = histogram(chain[:b1], bins=50, title="Posterior of b1", label="")
# p3 = histogram(chain[:W2], bins=50, title="Posterior of W2", label="")
# p4 = histogram(chain[:b2], bins=50, title="Posterior of b2", label="")

# # Combine the plots into a single figure
# plot(p1, p2, p3, p4, layout=(2, 2), legend=false)




# # Predict the next month's FOB value
# x_new = ts[end]
# y_pred = predict(bayesian_nn(input_size, hidden_size, output_size), x_new, posterior)
# println("Predicted next month's FOB value: $y_pred")

import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('EXPORT_2015_monthly_annual.csv')

# Group by MONTH and aggregate FOB values
monthly_fob = df.groupby('MONTH')['FREE ON BOARD (value in USD)'].sum()


# Prepare the dataset for training
X = monthly_fob.index.astype(float).values.reshape(-1, 1)  # Months as input
y = monthly_fob.values.reshape(-1, 1)  # FOB values as target

# Normalize the input
X_mean, X_std = X.mean(), X.std()
X_normalized = (X - X_mean) / X_std

# Define the prior and posterior for the weights
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(
        tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1.0),
        reinterpreted_batch_ndims=1)

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda t: tfp.layers.VariableLayer(2 * n, dtype=dtype)(t)),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# Define whether the prior is trainable or not
prior_trainable = True  # Set to True if you want the prior to be trainable

# Build a Bayesian neural network model
model = tf.keras.Sequential([
    tfp.layers.DenseVariational(10, posterior(10, prior_trainable), prior(10, prior_trainable),
                                 activation='relu'),
    tfp.layers.DenseVariational(1, posterior(1, prior_trainable), prior(1, prior_trainable))
])
# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model (you can add validation data if available)
model.fit(X_normalized, y, epochs=1000, verbose=0)

# Predict the next month's FOB value
next_month = X_normalized[-1] + 1  # Assuming next month is the next integer value
predicted_fob = model.predict(next_month)

# Denormalize the predicted value
predicted_fob = predicted_fob * X_std + X_mean

# Plot the predictions
plt.plot(X, y, label='Actual')
plt.axvline(x=next_month, linestyle='--', color='r', label='Next Month')
plt.scatter([next_month], predicted_fob, color='r')
plt.xlabel('Month')
plt.ylabel('FOB (USD)')
plt.legend()
plt.show()
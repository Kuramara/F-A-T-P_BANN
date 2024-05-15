import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp

# Step 1: Data Collection
data = pd.read_csv('agricultural_trade_data.csv')
features = data[['feature1', 'feature2', 'feature3']]
targets = data['trade_value']

# Step 2: Data Preprocessing
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Step 3: Feature Selection
# Assuming features have already been selected

# Step 4: Model Architecture and Step 5: Bayesian Inference
class BayesianNN(tf.keras.Model):
    def __init__(self):
        super(BayesianNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = BayesianNN()

def negloglik(y, y_pred):
    return -tf.reduce_mean(y_pred.log_prob(y))

# Priors
prior = tfp.distributions.Normal(loc=0., scale=1.)

# Convert to probabilistic layers
model.dense1 = tfp.layers.DenseFlipout(10, activation='tanh', kernel_prior_fn=lambda _: prior, bias_prior_fn=lambda _: prior)
model.dense2 = tfp.layers.DenseFlipout(1, kernel_prior_fn=lambda _: prior, bias_prior_fn=lambda _: prior)

# Compile model
model.compile(optimizer=tf.optimizers.Adam(), loss=negloglik)

# Step 6: Training
model.fit(X_train, y_train, epochs=1000, verbose=False)

# Step 7: Validation and Step 8: Prediction
y_pred = model(X_test)
y_pred_mean = y_pred.mean()
y_pred_std = y_pred.stddev()

# Step 9: Evaluation
mse = tf.reduce_mean(tf.square(y_pred_mean - y_test))
print(f'Mean Squared Error: {mse.numpy()}')

uncertainty = y_pred_std.numpy()
print(f'Uncertainty: {uncertainty}')

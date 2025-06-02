import pandas as pd
from keras.models import load_model

# Load model and proposed product data
model = load_model('trained_model.h5')
X_proposed = pd.read_csv("Module2/datasets/proposed_new_product.csv").values

# Predict and rescale
prediction_scaled = model.predict(X_proposed)[0][0]

# Replace SCALE and MIN with values from scaling step
SCALE = 0.0000034013  # Example value from scaling output
MIN = 0.000000        # Example value from scaling output

prediction = (prediction_scaled - MIN) / SCALE
print(f"Predicted Earnings: ${prediction:,.2f}")

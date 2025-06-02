import numpy as np
import scipy.io
import os

# Ensure the ./read_data directory exists
os.makedirs('./read_data', exist_ok=True)

# Create a dummy ts_train variable
# Shape (30000, 2) with random double values between 0 and 1
ts_train_dummy = np.random.rand(30000, 2)

# Save it to lorenz.mat
mat_file_path = './read_data/lorenz.mat'
scipy.io.savemat(mat_file_path, {'ts_train': ts_train_dummy})

print(f"Dummy file '{mat_file_path}' created successfully with 'ts_train' variable shape {ts_train_dummy.shape}.")

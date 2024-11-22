# Multidimensional Density
This function is used to calculate the binned density similar to `np.histogram(., density=True)` not just for one dimension, but for multiple in parallel.
The input `samples` is assumed to have the shape: \[n_samples, n_dimensions\]
The output will be:
    - `sample_densities`: The density of each sample (shape: \[n_samples, n_dimensions\])
    - `density`: The density for each dimension (shape: \[n_bins + 1, n_dimensions\])
    - `bins`: The equally distanced bins edges, controlled by `n_bins`(shape \[n_bins + 1, n_dimensions\])

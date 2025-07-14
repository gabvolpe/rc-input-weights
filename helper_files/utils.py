import numpy as np

def create_weights(shape: tuple, method: str, threshold: float = 1e-3):
    """
    Excludes any values below the absolute threshold.
    """
    if method == "random_uniform":
        w = np.random.uniform(-1, 1, size=shape)
        while np.any(np.abs(w) < threshold):
            idx = np.abs(w) < threshold
            w[idx] = np.random.uniform(-1, 1, size=np.sum(idx))
        return w
    
    elif method == "random_normal":
        w = np.random.randn(*shape)
        while np.any(np.abs(w) < threshold):
            idx = np.abs(w) < threshold
            w[idx] = np.random.randn(np.sum(idx))
        return w

    elif method == "laplace":
        w = np.random.laplace(loc=0.0, scale=0.5, size=shape).flatten()
        while np.any(np.abs(w) < threshold):
            idx = np.abs(w) < threshold
            w[idx] = np.random.laplace(loc=0.0, scale=0.5, size=np.sum(idx))
        return w.reshape(shape)

    elif method == "fourier":
        num_weights = shape[0]
        max_freq = 10
        t = np.linspace(0, 1, num_weights).reshape(-1, 1)
        frequencies = np.random.uniform(0, max_freq, size=(num_weights, 1))
        phases = np.random.uniform(0, 2*np.pi, size=(num_weights, 1))
        weights = np.sin(2*np.pi*frequencies*t + phases) + np.cos(2*np.pi*frequencies*t + phases)
        weights = 0.5 * weights / np.max(np.abs(weights))
        weights = weights.flatten()

        # Fix: use np.where to get indices where weights are small
        while np.any(np.abs(weights) < threshold):
            idxs = np.where(np.abs(weights) < threshold)[0]

            # Regenerate frequencies and phases for these indices only
            frequencies[idxs, 0] = np.random.uniform(0, max_freq, size=len(idxs))
            phases[idxs, 0] = np.random.uniform(0, 2*np.pi, size=len(idxs))

            # Recompute weights at these indices
            # Since t is shape (num_weights, 1), we can index t[idxs, 0] to get scalar t values for each idx
            weights_temp = (
                np.sin(2*np.pi*frequencies[idxs, 0] * t[idxs, 0] + phases[idxs, 0]) +
                np.cos(2*np.pi*frequencies[idxs, 0] * t[idxs, 0] + phases[idxs, 0])
            )
            # Normalize partial weights by max absolute weight from full weights array
            max_abs = np.max(np.abs(weights))
            weights[idxs] = 0.5 * weights_temp / max_abs

        return weights.reshape(shape)

    else:
        raise ValueError(f"Unknown weight initialization method: {method}")
    

def compute_median_and_iqr_loss(losses: list):
    numeric_losses = np.array([x for x in losses if isinstance(x, (int, float, np.float32, np.float64))])
    if len(numeric_losses) == 0:
        return None, None
    median = np.median(numeric_losses)
    q75, q25 = np.percentile(numeric_losses, [75, 25])
    iqr = q75 - q25
    return median, iqr
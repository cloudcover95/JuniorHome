# src/physiomanifold/recursive_feedback/active_inference.py

import mlx.core as mx

class FreeEnergyMinimizer:
    """
    Executes the Active Inference loop by minimizing Variational Free Energy.
    Bypasses standard backpropagation in favor of physical state-updating.
    """
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def compute_free_energy(self, internal_state: mx.array, sensory_input: mx.array, generative_weights: mx.array) -> mx.array:
        """
        Free Energy (F) = Complexity (KL Divergence proxy) + Inaccuracy (Prediction Error).
        Calculated directly via MLX tensor ops.
        """
        # 1. Generate prediction from internal state
        prediction = mx.matmul(internal_state, generative_weights)
        
        # 2. Inaccuracy: Mean Squared Error (Surprise)
        prediction_error = sensory_input - prediction
        inaccuracy = mx.mean(mx.square(prediction_error))
        
        # 3. Complexity: L2 norm of the internal state (Thermodynamic constraint)
        complexity = mx.mean(mx.square(internal_state)) * 0.1
        
        # Total Variational Free Energy
        return inaccuracy + complexity

    def execute_perception_step(self, internal_state: mx.array, sensory_input: mx.array, generative_weights: mx.array) -> mx.array:
        """
        Updates the internal state to minimize Free Energy via MLX automatic differentiation.
        """
        # Define the loss function with respect to the internal state
        def loss_fn(state):
            return self.compute_free_energy(state, sensory_input, generative_weights)
        
        # Compute gradient of Free Energy with respect to internal state
        loss, grad = mx.value_and_grad(loss_fn)(internal_state)
        
        # State update (Gradient Descent on Free Energy)
        updated_state = internal_state - (self.lr * grad)
        return updated_state, loss

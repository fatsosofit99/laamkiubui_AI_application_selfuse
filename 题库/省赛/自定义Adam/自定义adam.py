import numpy as np

def func(x: np.ndarray) -> np.ndarray:
    #TODO
    return np.sin(x**2) / x
def grad_func(x: np.ndarray) -> np.ndarray:
    #TODO
    return 2 * np.cos(x**2) - np.sin(x**2) / (x**2)
class CustomAdam:
    def __init__(self, params: np.ndarray, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.t = 0

    def update(self, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return self.params
        #TODO

def optimize_function_with_adam(initial_x: np.ndarray, iterations: int) -> None:
    params = initial_x
    adam = CustomAdam(params)
    for i in range(iterations):
        grads = grad_func(params)
        params = adam.update(grads)

        if i % 200 == 0 or i==iterations - 1:
            print(f"Iteration {i}: x = {params}, f(x) = {func(params[0])}")


if __name__ == '__main__':
    optimize_function_with_adam(initial_x=np.array([0.8, 0.9, 1.0]), iterations=3000)
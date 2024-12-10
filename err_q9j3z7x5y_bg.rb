require 'nmatrix'

def l1_regularization(params, lambda)
  params.map { |p| lambda * p.abs }.sum
end

def l2_regularization(params, lambda)
  params.map { |p| 0.5 * lambda * p**2 }.sum
end

def adam(params, grads, moment1, moment2, beta1, beta2, epsilon)
  moment1 = beta1 * moment1 + (1 - beta1) * grads
  moment2 = beta2 * moment2 + (1 - beta2) * grads**2
  corrected_moment1 = moment1 / (1 - beta1**t)
  corrected_moment2 = moment2 / (1 - beta2**t)
  params -= alpha * corrected_moment1 / (NMatrix.sqrt(corrected_moment2) + epsilon)
  return params, moment1, moment2
end

def rmsprop(params, grads, cache, alpha, epsilon, decay_rate)
  cache = decay_rate * cache + (1 - decay_rate) * grads**2
  params -= alpha * grads / NMatrix.sqrt(cache + epsilon)
  return params, cache
end

def adadelta(params, grads, accumulated_grad, accumulated_delta, rho, epsilon)
  accumulated_grad = rho * accumulated_grad + (1 - rho) * grads**2
  delta_x = NMatrix.sqrt(accumulated_delta + epsilon) / NMatrix.sqrt(accumulated_grad + epsilon) * grads
  params -= delta_x
  accumulated_delta = rho * accumulated_delta + (1 - rho) * delta_x**2
  return params, accumulated_grad, accumulated_delta
end

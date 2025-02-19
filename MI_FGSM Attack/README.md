## **MI-FGSM - Momentum Iterative Fast Gradient Sign Method**
The momentum method is a technique for accelerating gradient descent algorithms by accumulating a velocity
vector in the gradient direction of the loss function across
iterations. The memorization of previous gradients helps to
barrel through narrow valleys, small humps and poor local
minima or maxima. The momentum method also shows
its effectiveness in stochastic gradient descent to stabilize
the updates.


### **Formula**
To generate a non-targeted adversarial example x* from
a real example x, which satisfies the L∞ norm bound, the velocity vector is defined as:

$$
g_{t+1} = \mu \cdot g_t^* + \frac{\nabla J_t(x_t, y)}{\| \nabla J_t(x_t, y) \|_1} ;$$

then update x*t+1 by applying the sign gradient as:

$$
x_{t+1}^* = x_t^* + \alpha \cdot \text{sign}(g_{t+1}) ;
$$

### **Algorithm steps**
1) Initialize parameters

-   Set α = ε / T (step size per iteration).
-   Set g₀ = 0 (momentum term).
-  t x₀* = x (initial adversarial example as the original image).

2) Iterate for T steps (T is the total number of iterations):

- Compute the gradient.
- Update the momentum term: Accumulate the velocity vector in the gradient direction.
- Update the adversarial example: Apply the sign gradient update.
-  Ensure the updated adversarial example remains within the valid range and perturbation limit.
-  Return the final adversarial example

### **Results**

Degradation in accuracy of adversarial example for MI_FGSM vs I_FGSM with number of iterations
![Untitled](https://github.com/user-attachments/assets/88832fab-503b-4a8a-873b-7c3d0ff5eb33)

![IFGSM_vs_MIFGSM_1_acc=41 7 69 49 45](https://github.com/user-attachments/assets/58664ebd-91c9-4291-a0e6-164323c7976a)

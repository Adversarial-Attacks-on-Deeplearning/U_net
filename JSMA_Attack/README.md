
# Jacobian-based Saliency Map Attack
The JSMA is a method for crafting adversarial samples against deep neural networks (DNNs) 
introduced in the paper "The Limitations of Deep Learning in Adversarial Settings" by Nicolas Papernot et al. 
The attack leverages the forward derivative of the DNN to construct adversarial saliency maps, 
which identify the most influential input features to perturb in order to cause misclassification. 
In our work we tried to adapt it to image segmentation model with modifications to improve performance and 
adversarial effectiveness.

## Forward Derivative (Jacobian Matrix)
The Jacobian matrix JF(X)/JF​(X) represents how the model’s output changes with respect to small changes in the input. 
It helps identify which pixels/features have the greatest impact on the model’s decision. 

The jacobian matrix of the model's output with respect to the input image is defined as:

$$
J_F(X) = \frac{\partial F(X)}{\partial X} = 
\begin{bmatrix}
\frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_n} \\
\frac{\partial F_2}{\partial x_1} & \cdots & \frac{\partial F_2}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_n}
\end{bmatrix}
$$

where:
- $F(X)$ is the model's output (e.g., segmentation mask in JSMA).
- $X$ is the input image.
- each entry $\frac{\partial F_i}{\partial x_j}$ represents the change in the $i$-th output w.r.t the $j$-th input pixel.

To improve performance and speed, we instead calculate the element-wise gradients of the output of the sigmoid function w.r.t input image.

The element-wise gradient matrix is defined as:

for a given output pixel $y_{i,j}$ in the model's prediction:

$$
grad_{i,j} = \frac{\partial F_{i,j}}{\partial x_{i,j}}
$$

where $\text{grad}_{i,j}$ has the same shape as $X$. this represents how sensitive each output pixel is to changes in the input.

##  Adversarial Saliency Map
A saliency map is computed from the Jacobian to determine which input features should be perturbed to maximize the 
probability of an incorrect classification. The goal is to find pixels that, when modified, 
push the model towards the target class. To perform this, we use a tagret mask to define which pixels to keep thier gradients.


## Threat Model
We apply a targeted misclassification attack that Forces the DNN to classify the input into a specific target class. 
In the case of binary image segmentation it tries to make the model predict some pixels as class 1 or 0 depending on the targeted mask.

## Algorithm steps: 
The JSMA attack iteratively perturbs input features based on the adversarial saliency map until the DNN misclassifies the input into the target class or a maximum distortion threshold is reached.


  1) Compute the element-wise gradients of the output w.r.t input image
  2) Construct the adversarial saliency map based on the targeted mask
  3) Select the most salient feature(s) to perturb).
  4) Modify the selected feature(s) by θ.
  5) Repeat until the distortion limit is reached.

## Results

Targeted mask used:

![targeted_mask](https://github.com/user-attachments/assets/06713b28-e39b-4b45-8fef-0c954aa554fd)

1) Setting theta = 1, and maximum total distribution = 4.15% (The minimum achieved in Papernot et al)
![JSMA](https://github.com/user-attachments/assets/894e2a68-e65d-4f1b-9d93-11b48970c369)

2) Setting theta = 0.05 and maximum total distribution = 20%, while adding limit on distribution per pixel = 12.549%
![JSMA_mod2](https://github.com/user-attachments/assets/7cd852ef-5926-452c-9536-99b6e3c89f54)



  
  
  

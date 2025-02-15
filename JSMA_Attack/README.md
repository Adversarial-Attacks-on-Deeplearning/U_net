
****Not finished***
# Jacobian-based Saliency Map Attack
The JSMA is a method for crafting adversarial samples against deep neural networks (DNNs) 
introduced in the paper "The Limitations of Deep Learning in Adversarial Settings" by Nicolas Papernot et al. 
The attack leverages the forward derivative of the DNN to construct adversarial saliency maps, 
which identify the most influential input features to perturb in order to cause misclassification. 
In our work we tried to adapt it to image segmentation model with modifications to improve performance and 
adversarial effectiveness.

## Forward Derivative (Jacobian Matrix)
The Jacobian matrix JF(X)/JF​(X) represents how the model’s output changes with respect to small changes in the input. 
It helps identify which pixels/features have the greatest impact on the model’s decision. To improve performance and speed, 
we instead calculate the element-wise gradients of the output of the sigmoid function w.r.t input image.
```
def compute_jacobian(model, image):
    """ Compute the Jacobian matrix of the model output with respect to the input image. """

    image.requires_grad = True
    output = torch.sigmoid(model(image.unsqueeze(0)))  # Forward pass
    grad_outputs = torch.ones_like(output.squeeze(0))  # Same shape as output
    element_wise_grad = torch.autograd.grad(output.squeeze(0), image, grad_outputs=grad_outputs, create_graph = True)[0]
    return element_wise_grad
```


##  Adversarial Saliency Map
A saliency map is computed from the Jacobian to determine which input features should be perturbed to maximize the 
probability of an incorrect classification. The goal is to find pixels that, when modified, 
push the model towards the target class. To perform this, we use a tagret mask to define which pixels to keep thier gradients.
```
def saliency_map(jacobian, target_label):
    """ Compute the saliency map to determine which pixels to perturb. """
    # Compute impact of modifying each pixel
    target_grad = jacobian * target_label  # Positive if it pushes towards target class    
    return target_grad  # Saliency scores
```

## Threat Model
We apply a targeted misclassification attack that Forces the DNN to classify the input into a specific target class. 
In the case of binary image segmentation it tries to make the model predict some pixels as class 1 depending on the targeted mask.

## Algorithm: 
The JSMA attack iteratively perturbs input features based on the adversarial saliency map until the DNN misclassifies the input into the target class or a maximum distortion threshold is reached.

Steps:

  1) Compute the element-wise gradients of the output w.r.t input image
  2) Construct the adversarial saliency map based on the targeted mask
  3) Select the most salient feature(s) to perturb).
  4) Modify the selected feature(s) by an initial small amount θ.
  5) Repeat until the requered accuracy or the distortion limit is reached.

  
  
  

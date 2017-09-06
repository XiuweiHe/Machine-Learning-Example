# Machine Learning Example
# 2.Training Machine Learning Algorithms for Classification
## 1.Artificial Neurons
* input values *x*,weights vector *w*,*activation function* $\phi(z)$, where z is the so-called net input($z = w_1x_1 + w_2x_2 + ...+ w_mx_m$):$$ w= \begin{bmatrix} w_1\\ w_2\\ \vdots \\ w_m \end{bmatrix},\; x= \begin{bmatrix} x_1\\x_2\\ \vdots \\x_m \end{bmatrix}  $$
* Heaviside （海威塞德）step fuction $ \phi(\cdot)$:
$$ \phi(z)=\begin{cases} 1,& \text{ $z \geq \theta$} \\ -1, & \text {otherwise} \end{cases} $$
For simplicity,we can bring the threshold $\theta$ to the left side of equation and define a weight-zero as $w_0 = -\theta$ and $x_0 = 1$,so that we can write z in a compact form $z=w_0x_0+ w_1x_1 +\cdots + w_mx_m =\sum_{j=0}^m x_jw_j = \bf{w}^T\bf{x}$ and $ \phi(z)=\begin{cases} 1,& \text{ $z \geq \theta$} \\ -1, & \text {otherwise} \end{cases} $

## 2.Perceptron Classifier Implementation Step
1. Initialize the weights to 0 or samll random numbers.
2. For each training samples $x^{(i)}$ perform the following steps:
	1. Compute the output value $\hat{y}$.
	2. Update the weights.

**Note**: 
1. $\hat{y}$ is the class label predicted by the perceptron classifier.
2. $w_j := w_j + \Delta w_j$,  where $\Delta w_j =\eta (y^{(i)}-\hat{y}^{(i)})x^{(i)}_{j}$, $\eta$ is the learning rate (a constant between 0.0 and 1.0)
3. when the true class label same as the predicted class label, weights update value are zero, the weight update is proportional to the value of $x^{(i)}_{j}$
4. the **convergence** of perceptron is guaranteed by the two classes are **linearly seperable** and **learning rate is sufficiently small**. if two classes can't be seperated by a linear decision boundary ,we can set a maximun number of passes over the training dataset(*epochs*) and /or a threshold for the number of tolerated misclassifications -the perceptron would never stop updating the weights otherwise

## 3.Adaptive linear neurons and the convergence of learning
**Adaptive linear neuron(Adaline)** illustrates the key concept of defining and minimizing cost of functions,which will lay the groundwork for understanding more advanced machine learning algorithms for classification,such as logistic regression and support vector machincs.
### 1.Minimizing cost functions with gradient descent
 *objective function (cost function)* is one of the key ingredients of supervised machine learning algorithms that is to be optimized during the learning process.In case of Adaline ,we can define the cost funciton$J$ to learn the weights as the **Sum of Squared Errors(SSE)** between the calculated outcome and the ture class label.$$J(w) = \frac{1}{2}\sum_{i}(y^{(i)}-\phi(z^{(i)}))^2$$
the main advantage of this continuous linear activation function is -in contrast to the unit step function- that the cost function becomes **differentiable**.and nice property of this cost funciton is that it is convex.thus,we can use *gradient descent* to find the weights that miniminze our cost funcion to classifiy the samples in the Iris dataset.
+ Gradient descent algorithm
 we can describe the principle behind gradient descent as *climbing down a hill* until a local or global cost minimun is reached.In each iteration,the step size determined by the value of the learning rate as well as the slope of the gradient:
+ Gradient descent step:

	* step1:update the weights by taking away from the gradient $\nabla J(w)$ of our cost function$J(w)$:
$$ w := w + \Delta w$$
Here,the weight change $\Delta w$ is defined as the negative gradient multiplied by the learing rate $\eta$:
$$ \Delta w = -\eta \Delta J(w)$$
To compute the gradient of the cost function,we need to compute the partial derivative of the cost function with respect to each weight $w_j$:
$$ \frac{\partial J}{\partial w_j} = -\sum_{i}(y^{(i)}-\phi(z^{(i)})x^{(i)}_{j}$$
$$ \Delta w_j= -\eta \frac{\partial j}{\partial w_j} = \mu \sum_{i}(y^{(i)}-\phi(z^{(i)})x^{(i)}_{j}$$
+ Result analysis:

	* choose a learning rate too large($\eta = 0.01$)-instead of minimizing the cost function,the errror becomes larger in every epoch because we overshoot the global minimum.
	* choose a learing rate too small($\eta = 0.0001$) cause that the algorithm would require a very large number of epochs to converge.

### 2.Feature scaling
Use feature scaling to get optimal performance.Gradient descent is one of the many algorithms that benefit from feature scaling.Here,we will use a feature scaling method called *standardization*,which gives our data the property of a standard normal distribution.The mean of each feature is centered at value 0 and the feature column has a standard deviation of 1.For example,to standardize the $j$th feature,we simply need to subtract the sample mean $\mu _j$ from every training sample and divide it by its standard devtiaion $\sigma _j$: $$ x^{\prime}{j}=\frac{x_j-\mu _j}{\sigma _j}$$
Here $x_j$ is a vector consisting of the $j$th feature values of all training samples $n$.Standardization can easily be achieved using the NumPy methods *mean* and *std*.
* Result analysis:
	
	*we still use Adaline method to classify the Iris dataset labels,after processing the original data features to standardized features,we use a learning rate $\eta =0.01$ can make the GD quicky converged at about 15 epochs.

### 3.Large scale machine learning and stochastice gradient descent
In the previous section,we used the gradient descent method called *bath gradient descent* that has a diadvantage is required large computation in the very large dataset with millions of data points.To address the problem,a popular alternative method is*stochastic gradient descent*.Instead of updating the weights based on the sum of the accumulated errors over all samples $x^{(i)}$:
$$ \Delta w= \eta \sum_{i}(y^{(i)}-\phi(z^{(i)})x^{(i)} \, ,$$
We update the weights incrementally for each training sample:
$$\eta (y^{(i)}-\phi(z^{(i)})x^{(i)}$$
Although stochastic gradient descent can be considered as an approximation of gradient descent,it typically reaches convergence much faster because of the **more frequent weight updates**.Since each gradient is calculated based on a single training example,the **error surface is noisier** than in gradient descent,which can also have the advantage that stochastic gradient descent can **escape shallow local minima** more readily.To obtain accurate results via stochastic gradient descent,it is important to present it with data in a random order,which is why we want to shuffle the training set for every epoch to prevent cycles.

**Note:**
>In stochastic gradient descent implementations,the fixed learning rate $\eta$ is often replaced by an adaptive learning rate that decreases over times, for example,$\frac{c_1}{[number\, of\, iterations]+c_2}$where $c_1$ and $c_2$ are constants.Note that stochastic gradient descent does not reach the global minimum but an area very close to it.By using an adaptive learning rate,we can ahieve furtherannealing to a better global minimum

Another advantage of stochastic gradient descent is that we can use it for *online learning*.In online learning,our model is trained on-the-fly as new training data arrives.This is especially useful if we are accumulating large amounts of data-for example,customer data in typical web applications.Using online learning,the system can immediately adapt to changes and the training data can be discarded after updating the model if storage space in an issue.

**Note:**
>A compromise between batch gradient descent and stochastic gradient descent is the so called *mini-bath learning*.Mini-batch learning can be understood as applying batch gradient descent to smaller subsets of the training data-for example,50 samples at a time.The advantage over batch gradient descent is that convergence is  reached faster via mini-batches because of the more frequent weight updates.Furthermore,mini-batch learning allows us to replace the for-loop over the training samples in **stochastic gradient descent(SGD)** by vectorized operations,which can further improve the computational efficiency of our learning algorithm.

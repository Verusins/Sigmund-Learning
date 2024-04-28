This is freeCodeCamp ML learning nodes

**Tensorflow Tutorial**

**Machine learning Fundamental** 1/32
Artificial Intelligence - A.I.

- A.I. can be _predefined_: e.g. tic-tac-toe / chess

  (It is a bunch of rules been created for the A.I. , could be simple)
  more complex example: e.g. pacman: facing the character and chase

ML is a part of AI,

> **Machine Learning** is _figuring out the rules_ for us, replacing the predefined rules.

ML Goal: raise accuracy
Feeding answers to the model instead of rules.

> **Neural Networks** is a part of Machine Learning, its _uses a **layered** representation of data_

- Data:
  - Features: input
  - Label: output
    using the inputs to predict the output form the given data.

**Introduction to TensorFlow** 2/32

> a **tensor** is a generalization of vectors and matrices in a higher dimension, a type of data point. Each tensor represents a partially defined computation that will eventually produce a value.

**Shape**: represents the dimension of data.

Create Tensors: e.g.

```
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(114514, tf.int16) (swap w .float64)
rank1_tensor = tf.Variable(["test", "ok", "burger"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
```

(This is where the higher dimension kicks in!)

Determine rank of tensor:
`tf.rank(rank2_tensor)`

-> <tf.Tensor: shape=(), dtype=int32, numpy=1>

Determine the shape of tensor / reshape:

`rank2_tensor.shape`

-> Tensorshape([2, 2]) # first 2 for interior list len, second 2 for outer

`rank1_tensor.shape`

-> Tensorshape([3])

````tensor1 = tf.ones(1,2,3) # creates a shape of [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2,3,1]) # reshaping
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 for auto calculate, reshaping to [3,3]
# auto calculate: e.g. we have 500 elements reshaping to [5, -1] would lead to [5, 100]

print(tensor1)
print(tensor2)
print(tensor3)```

output:

```tf.Tensor([[[1. 1. 1.][1. 1. 1.]]], shape=(1, 2, 3), dtyp2=float32)
tf.Tensor...```

Types of Tensors: (Above are variables)

- **Variable**
- **Constant**
- Placeholder
- SparseTensor

Evaluate Tensor:
```with tf.Session() as sess: # sess representing any session
    tensor.eval() # tensor auto determining which course is used to evaluate```


**Core Learning Algorithms A** 3/32
TensorFlow Core Learning Algors
- Linear Regression
- Classificatin
- Clustering
- Hiddern Markov Models

**Linear Regression**: can be in 3D, need more coordinates than y=mx+b
e.g. in R^3, we use x,y to predict z, or x,z to predict y.
`!pip install -q sklearn`
and a ton of imports, refer to the documentation
`import numpy as np` an optimized array in python, like cross product etc.
`import pandas as pd` a data manipulation tool (visualize / cut rows)
`import matplotlib.pyplot as plt` visualize graph and dataset
`import tensorflow.compat.v2.feature_column as fc` THE CORE

**Core Learning Algorithms B** 4/32

a example dataset: titanic dataset
"Whos going to survive given the information"
Load dataset using `pd.read_csv` and `dftrain.pop`
Linear regression is a good fit because the parameters directly affect the person's surviving the titanic crash
````

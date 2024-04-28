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

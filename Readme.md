# branch v1.0
fix problem when tensorflow v1.0 is used.
```
diff --git a/gan.py b/gan.py
index e37ba07..d84fc9c 100644
--- a/gan.py
+++ b/gan.py
@@ -56,7 +56,9 @@ def linear(input, output_dim, scope=None, stddev=1.0):
 
 
 def generator(input, h_dim):
-    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
+    # fix tensorflow v1.0
+    # h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
+    h0 = tf.log(tf.exp(linear(input, h_dim, 'g0')) + 1)
     h1 = linear(h0, 1, 'g1')
     return h1
 
@@ -82,7 +84,9 @@ def minibatch(input, num_kernels=5, kernel_dim=3):
     diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
     abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
     minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
-    return tf.concat(1, [input, minibatch_features])
+    # fix tensorflow v1.0
+    # return tf.concat(1, [input, minibatch_features])
+    return tf.concat([input, minibatch_features], 1)
```

# An introduction to Generative Adversarial Networks

This is the code that we used to generate our GAN 1D Gaussian approximation.
For more information see our blog post: [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow).

## Installing dependencies

Written for Python 2.7.x.

For the Python dependencies, first install the requirements file:

    $ pip install -r requirements.txt

You should then install TensorFlow `0.12`, see: [https://www.tensorflow.org/get_started/os_setup#pip_installation](https://www.tensorflow.org/get_started/os_setup#pip_installation).

If you want to also generate the animations, you need to also make sure that `ffmpeg` is installed and on your path.

## Training

For a full list of parameters, run:

    $ python gan.py --help

To run without minibatch discrimination (and plot the resulting distributions):

    $ python gan.py

To run with minibatch discrimination (and plot the resulting distributions):

    $ python gan.py --minibatch True

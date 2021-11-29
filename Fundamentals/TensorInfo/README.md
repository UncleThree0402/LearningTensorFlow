# Tensor Info

<pre>
Check rank with : tensor.ndim
Check shape with : tensor.shape
Check shape with specific axis : tensor.shape[n]
Check type with : tensor.dtype
Check size with : tf.size(tensor)
*For returning numpy value can add '.numpy()' at the end
</pre>

<pre>
Finding index of tensor is same with python list
Resize with tensor[..., tf.newaxis] 
Or tf.expand_dims(tensor,axis=n)

Squeezing Tensor:
remove all axis with value one
tf.squeeze(tensor)
</pre>
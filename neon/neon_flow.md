alxnet.py:

[TOC]
# Flow Analysis with Alexnet

## Background

[Neon Design Decisions](http://neon.nervanasys.com/docs/latest/design.html)
[Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
[Neon code](https://github.com/nervanasystems/neon)

## fprop

### Common path

`model::benchmark() ==> model::fprop() ==> Sequential::fprop() ==> xxx_layer::frop()`

* model.benchmark(train, cost=cost, optimizer=opt, niterations=10, nskip=5)

    x:   x is train = ArrayIterator(X_train, y_train, nclass=1000, lshape=(3, 224, 224))

    self: model

* x = self.fprop(x, inference)

``` python
    def benchmark(self, dataset, inference=False, cost=None, optimizer=None,
                  niterations=20, nskip=2):
         while count < niterations + nskip:
            dataset.reset()
            for mb_idx, (x, t) in enumerate(dataset):
                x = self.fprop(x, inference)
                if inference is False:
                    delta = self.cost.get_errors(x, t)
                    self.bprop(delta)
                    self.optimizer.optimize(self.layers_to_optimize, epoch=0)
                    self.be.record_mark(bprop_end)  # mark end of bprop
                    self.be.synchronize_mark(bprop_end)
                else:
                    self.be.synchronize_mark(fprop_end)

                count += 1
                if count >= niterations + nskip:
                    break
```

* `Model::fprop() ==> Sequential::fprop()`

```python
    def fprop(self, x, inference=False):
        """
        Forward propagates a minibatch x through the model.

        Arguments:
            x (Tensor): Input minibatch data.
            inference (bool): Flag for performing training or inference
                Only affects batch norm and dropout layers.

        Returns:
            Tensor: the output of the final layer in the model
        """
        return self.layers.fprop(x, inference)
```

* `Sequential::fprop() ==> [layers]::fprop()`

```python
    def fprop(self, inputs, inference=False, beta=0.0):
        x = inputs

        for l in self.layers:
            altered_tensor = l.be.distribute_data(x, l.parallelism)
            l.revert_list = [altered_tensor] if altered_tensor else []

            if l is self.layers[-1] and beta != 0:
                x = l.fprop(x, inference=inference, beta=beta)
            else:
                x = l.fprop(x, inference=inference)

        if inference:
            self.revert_tensors()

        return x
```
### Convolution layer fprop

`model::benchmark() ==> model::fprop() ==> Sequential::fprop() ==>Convolution::fprop() ==>  NervanaCPU::fprop_conv() ==> ConvLayer::xprop_conv()`



* `Sequential::fprop() ==> Convolution::fprop() ==> NervanaCPU::fprop_conv()`

* `Convolution::configure() ==> NervanaCPU::conv_layer() ==> ConvLayer::__init__()`

``` python
# CConvolution::configure()
def configure(self, in_obj):
           self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
```

``` python
# NervanaCPU::conv_layer()
   def conv_layer(self, dtype,
                   N, C, K,
                   D=1, H=1, W=1,
                   T=1, R=1, S=1,
                   pad_d=0, pad_h=0, pad_w=0,
                   str_d=1, str_h=1, str_w=1):
        return ConvLayer(self, dtype, N, C, K, D, H, W, T, R, S,
                         pad_d, pad_h, pad_w, str_d, str_h, str_w)

```

```python
    def fprop(self, inputs, inference=False, beta=0.0):
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, 
                            beta=beta, bsum=self.batch_sum)
        return self.outputs
```

* `Convolution::fprop() ==> NervanaCPU::fprop_conv() ==> ConvLayer::xprop_conv()`

```python
    def fprop_conv(self, layer, I, F, O,
                   X=None, bias=None, bsum=None,
                   alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0):
        layer.xprop_conv(I, F, O, X, bias, bsum, alpha, beta, relu, brelu, slope)

```

```python
    def xprop_conv(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0, backward=False):

        if not backward:
            C, D, H, W, N = self.dimI
            C, T, R, S, K = self.dimF
            K, M, P, Q, N = self.dimO
            pad_d, pad_h, pad_w = self.padding
            str_d, str_h, str_w = self.strides

            I = I._tensor.reshape(self.dimI)
            O1 = O._tensor.reshape(self.dimO)
            O._tensor.resize(self.dimO)

            inPtr = c_longlong(I.ctypes.data)
            outPtr = c_longlong(O1.ctypes.data)
            weightPtr = c_longlong(F._tensor.ctypes.data)
            if bias:
                biasPtr = bias._tensor.ctypes.data
            else:
                biasPtr = 0
            primitives = c_longlong(self.dnnPrimitives.ctypes.data)
            self.mklEngine.SpatialConvolution_MKLDNN_forward(inPtr,outPtr,weightPtr,biasPtr,primitives,self.initOk,N,C,H,W,S,R,str_h,str_w,pad_w,pad_h,K,P,Q)
            self.initOk = 1
        else:
            if X is None:
                X = O

            if backward:
                I = I._tensor.reshape(self.dimO)
                O = O._tensor.reshape(self.dimI)
                X = X._tensor.reshape(self.dimI)
            else:
                I = I._tensor.reshape(self.dimI)
                O = O._tensor.reshape(self.dimO)
                X = X._tensor.reshape(self.dimO)
            F = F._tensor.reshape(self.dimF)
            if bias is not None:
                bias = bias._tensor.reshape((O.shape[0], 1))
            if bsum is not None:
                bsum = bsum._tensor.reshape((O.shape[0], 1))

```

### Full Connection Layer fprop (Linear::fprop())

`model::benchmark() ==> model::fprop() ==> Sequential::fprop() ==>Linear::fprop() ==>  NervanaCPU::compound_dot()`

```python
# Linear::fprop()
    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        if self.actual_bsz is None and self.actual_seq_len is None:
            self.be.compound_dot(A=self.W, B=self.inputs, C=self.outputs, beta=beta,
                                 bsum=self.batch_sum)
        else:
            bsz = self.be.bsz if self.actual_bsz is None else self.actual_bsz
            steps = self.nsteps if self.actual_seq_len is None else self.actual_seq_len

            self.be.compound_dot(A=self.W,
                                 B=self.inputs[:, :bsz * steps],
                                 C=self.outputs[:, :bsz * steps],
                                 beta=beta,
                                 bsum=self.batch_sum)

        return self.outputs
```


``` python
# NervanaCPU::compound_dot()
    def compound_dot(self, A, B, C, alpha=1.0, beta=0.0, relu=False, bsum=None):
        """
        Doing following operations (* is dot product)
        C = alpha * A * B   + beta * C
        C = alpha * A.T * B + beta * C
        C = alpha * A * B.T + beta * C.

        relu: if true applied before output (and prior to beta addition)

        The operation will be short-circuited to: out <- alpha * left * right
        if beta has value 0 (the default).

        Arguments:
            A, B (CPUTensor): input operands
            C (CPUTensor): output
            alpha (float): scale A*B term
            beta (float): scale C term before sum
            relu (bool): whether to apply ReLu before output
        """

        # checking type and shape
        assert A.dtype == B.dtype == C.dtype

        assert A.shape[0] == C.shape[0]
        assert B.shape[1] == C.shape[1]
        assert A.shape[1] == B.shape[0]

        # cleaner implementation, shall be equivalent to the one below
        # if relu:
        #     C[:] = self.log(1. + self.exp(alpha * self.dot(A, B))) + beta * C
        # else:
        #     C[:] = alpha * self.dot(A, B) + beta * C

        if beta == 0:
            if C._tensor.flags['C_CONTIGUOUS'] is not True:
                tmp = np.empty(C.shape, dtype=C.dtype)
                np.dot(A._tensor, B._tensor, tmp)
                C._tensor[:] = tmp.copy()
            else:
                np.dot(A._tensor, B._tensor, C._tensor)

            if relu:
                self.Relu(C._tensor, C._tensor)
        else:
            np.multiply(C._tensor, beta, C._tensor)
            tmp = np.empty(C.shape, dtype=C.dtype)
            np.dot(A._tensor, B._tensor, tmp)
            np.multiply(tmp, alpha, tmp)
            if relu:
                self.Relu(tmp, tmp)
            np.add(C._tensor, tmp, C._tensor)
        if bsum is not None:
            bsum[:] = self.sum(C, 1)

        return C

```




### Activation layer fprop

`model::benchmark() ==> model::fprop() ==> Sequential::fprop() ==>Activation::fprop() ==>  Reclin::__call__()`

* activation::fprop() ==> activation::transform() ==> `Reclin::__call__()`

``` python
# activation::fprop()
    def fprop(self, inputs, inference=False):
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs
```

``` python
# Reclin::__call__()
    def __call__(self, x):
        """
        Returns the Exponential Linear activation

        Arguments:
            x (Tensor or optree): Input value

        Returns:
            Tensor or optree: output activation
        """
        return (self.be.maximum(x, 0) + self.slope * self.be.minimum(0, x))
```

### Pooling layer fprop
`model::benchmark() ==> model::fprop() ==> Sequential::fprop() ==>Pooling::fprop()==> NervanaCPU::fprop_pool()`

``` python
# Pooling::fprop()
    def fprop(self, inputs, inference=False, beta=0.0):
        self.inputs = inputs
        self.be.fprop_pool(self.nglayer, inputs, self.outputs, self.argmax, beta=beta)
        return self.outputs
```

``` python
# NervanaCPU::fprop_pool()
    def fprop_pool(self, layer, I, O, argmax=None, beta=0.0):
        """
        Forward propagate pooling layer.

        Arguments:
            layer (PoolLayer): The pool layer object, different backends have
                               different pool layers.
            I (Tensor): Input tensor.
            O (Tensor): output tensor.
            argmax (Tensor): tensor to store location of the maximum
        """

        assert layer.sizeI == I.size
        assert layer.sizeO == O.size
        if layer.op == "max":
            assert layer.sizeO == argmax.size
        op = layer.op

        J, T, R, S = layer.JTRS
        C, D, H, W, N = layer.dimI
        K, M, P, Q, N = layer.dimO
        pad_c, pad_d, pad_h, pad_w = layer.padding
        str_c, str_d, str_h, str_w = layer.strides

        array_I = I._tensor.reshape(layer.dimI)
        array_O = O._tensor.reshape(layer.dimO)
        primitives = c_longlong(layer.dnnPrimitives.ctypes.data)
        inPtr = c_longlong(I._tensor.ctypes.data)
        outPtr = c_longlong(O._tensor.ctypes.data)

        if op == "max":
            array_argmax = argmax._tensor.reshape(layer.dimO)

        for k in range(K):
            sliceC, _ = layer.kSlice[k]

            for m in range(M):
                sliceD, _ = layer.mSlice[m]

                for p in range(P):
                    sliceH, _ = layer.pSlice[p]

                    for q in range(Q):
                        sliceW, _ = layer.qSlice[q]

                        sliceI = array_I[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                        if op == "max":
                            array_argmax[k, m, p, q, :] = np.argmax(sliceI, axis=0)
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + \
                                np.max(sliceI, axis=0)
                        elif op == "avg":
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + \
                                np.mean(sliceI, axis=0)
                        elif op == "l2":
                            array_O[k, m, p, q, :] = array_O[k, m, p, q, :] * beta + \
                                np.sqrt(np.sum(np.square(sliceI), axis=0))
```


## Cost

### GeneralizedCost

`Model::benchmark() ==> GeneralizedCost::get_cost() ==> CrossEntropyMulti::__call__()`

``` python
# GeneralizedCost::get_cost()
    def get_cost(self, inputs, targets):
        """
        Compute the cost function over the inputs and targets.

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.

        Returns:
            Tensor containing cost

        """
        self.outputs[:] = self.costfunc(inputs, targets)
        self.be.mean(self.outputs, axis=1, out=self.cost_buffer)
        self.cost = self.cost_buffer.get()
        return self.cost
```


## Error

### CrossEntropyMulti

`Model::benchmark() ==> GeneralizedCost::get_errors() ==> CrossEntropyMulti::bprop()`

 * Model::benchmark()

```python
                    delta = self.cost.get_errors(x, t)
                    self.bprop(delta)
```

 * GeneralizedCost::get_errors()

```python
# GeneralizedCost::get_errors()
    def get_errors(self, inputs, targets):
        """
        Compute the derivative of the cost function

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.

        Returns:
            Tensor of same shape as the inputs containing their respective
            deltas.
        """
        self.deltas[:] = self.costfunc.bprop(inputs, targets)
        return self.deltas
```

  * CrossEntropyMulti::bprop()

```python
    def bprop(self, y, t):
        """
        Returns the derivative of the multiclass cross entropy cost.

        Args:
            y (Tensor or OpTree): Output of previous layer or model
            t (Tensor or OpTree): True targets corresponding to y

        Returns:
            OpTree: Returns the (mean) shortcut derivative of the multiclass
            entropy cost function ``(y - t) / y.shape[1]``
        """
        return self.scale * (y - t)
```


## bprop
### Common path
#### Model::bprop()

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> XXX_Layer::bprop()`

```python 
# Model::bprop()
    def bprop(self, delta):
        """
        Back propagates the error of a minibatch through the model.

        Arguments:
            delta (Tensor): Derivative of cost with respect to the last layer's output

        Returns:
            Tensor: Deltas to propagate to the next layer
        """
        return self.layers.bprop(delta)
```

#### Sequential::bprop()

`Sequential::bprop() ==> Convolution::bprop()`

``` python
#Sequential::bprop()
    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        for l in reversed(self._layers):
            altered_tensor = l.be.distribute_data(error, l.parallelism)
            if altered_tensor:
                l.revert_list.append(altered_tensor)
            if type(l.prev_layer) is BranchNode or l is self._layers[0]:
                error = l.bprop(error, alpha, beta)
            else:
                error = l.bprop(error)

            for tensor in l.revert_list:
                self.be.revert_tensor(tensor)
        return self._layers[0].deltas
```

### Activation Layer bprop

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> Activation::bprop()`


```python
# Activation::bprop()
    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        self.deltas[:] = self.transform.bprop(self.outputs) * error
        return self.deltas
```

  * l: Activation Layer 'Linear_2_softmax': Softmax

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> Activation::bprop() ==> Softmax::bprop()`

``` python
# Softmax::bprop()
	def bprop(self, x):
        return 1
```

  * l: Activation Layer 'Linear_1_Rectlin': Rectlin

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> Activation::bprop() ==> Rectlin::bprop()`

```python
# Rectlin::bprop()
    def bprop(self, x):
		return self.be.greater(x, 0) + self.slope * self.be.less(x, 0)
```

### Convolution Layer bprop
  * ConvLayer::xprop_conv()

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> Convolution::bprop() ==> NervanaCPU::bprop_conv() ==> ConvLayer::xprop_conv()`

``` python
# Convolution::bprop()

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas,
                               alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW)
        return self.deltas
```

``` python
# NervanaCPU::bprop_conv()
    def bprop_conv(self, layer, F, E, grad_I,
                   X=None, bias=None, bsum=None,
                   alpha=1.0, beta=0.0,
                   relu=False, brelu=False, slope=0.0):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            F (CPUTensor): the weights (filters)
            E (CPUTensor): errors
            grad_I (CPUTensor): gradient to inputs (output delta)

        Compounding Options:
            X: tensor to use in bprop_relu or beta
                can be same as grad_I for beta accumulate (this is default when None)
                should be same shape as grad_I
            bias: (K,1) tensor to use for adding bias to output
                grad_I += bias
            bsum: (K,1) tensor to accumulate batch sum over (used in batchnorm or bprop_bias)
                bsum = sum(grad_I.reshape(K,-1), axis=1)
                the sum operation is fully deterministic
            alpha, beta:
                grad_I = alpha*grad_I + beta*X
                grad_I = alpha*grad_I + beta*grad_I   (if X==grad_I)
            relu, slope: boolean flag to apply:
                grad_I = max(grad_I, 0) + slope*min(grad_I, 0)
                can be combined with bias (where bias is added first)
            brelu, slope: boolean flag to apply:
                grad_I *= (X > 0) + slope*(X < 0)
                can be combined with bsum tensor to output bprop_bias
        """
        layer.xprop_conv(E, F, grad_I, X, bias, bsum, alpha, beta, relu, brelu, slope,
                         backward=True)
```


  * ConvLayer::update_conv()

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> Convolution::bprop() ==> NervanaCPU::update_conv() ==> ConvLayer::update_conv()`

```python
# NervanaCPU::update_conv()
	def update_conv(self, layer, I, E, U, alpha=1.0, beta=0.0):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            layer: the conv layer as a parameter object
            I (CPUTensor): the inputs
            E (CPUTensor): the errors
            U (CPUTensor): the updates
            alpha (float): linear scaling
            beta  (float): scaled accumulation
        """
        assert layer.sizeI == I.size
        assert layer.sizeO == E.size
        assert layer.sizeF == U.size

        layer.update_conv(I, E, U, alpha, beta)
```
``` python
# ConvLayer::update_conv()
    def update_conv(self, I, E, U, alpha=1.0, beta=0.0):

        C = self.C
        K, M, P, Q, N = self.dimO

        I = I._tensor.reshape(self.dimI)
        E = E._tensor.reshape(self.dimO)
        U = U._tensor.reshape(self.dimF)

        # 1x1 conv can be cast as a simple dot operation
        if self.dot:
            # CxK = CxHWN . KxHWN.T
            I = I.reshape((C, -1))
            E = E.reshape((K, -1)).T
            if beta:
                U[:] = alpha * np.dot(I, E).reshape(U.shape) + beta * U
            else:
                U[:] = alpha * np.dot(I, E).reshape(U.shape)
            return

        if beta:
            U *= beta
        else:
            U.fill(0.0)

        for m in range(M):
            sliceT, sliceD, tlen = self.mSlice[m]
            for p in range(P):
                sliceR, sliceH, rlen = self.pSlice[p]
                for q in range(Q):
                    sliceS, sliceW, slen = self.qSlice[q]

                    slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                    slicedE = E[:, m, p, q, :]
                    update = np.dot(slicedI, slicedE.T).reshape((C, tlen, rlen, slen, K))
                    if alpha == 1.0:
                        U[:, sliceT, sliceR, sliceS, :] += update
                    else:
                        U[:, sliceT, sliceR, sliceS, :] += alpha * update
```


### Linear Layer bprop
  * l: Linear Layer 'Linear_1'

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> Linear::bprop() ==> NervanaCPU::compound_dot() ==> np.dot()`

### Pooling layer bprop
  * l: Pooling_layer 'Pooling_2'

`Model::BenchMark()  ==> Model::bprop() ==> Sequential::bprop() ==> Pooling::bprop() ==> NervanaCPU::bprop_pool()`

## note

反复 {
    Sequential::frop() 遍历模型中各层，先依次计算每层的fprop（）
    计算error
    Sequential::brop() 遍历模型中各层，先依次计算每层的bprop（）
    Sequential::optimizer() 遍历模型中各层，并优化
}

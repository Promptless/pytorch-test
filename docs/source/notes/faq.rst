Frequently Asked Questions
==========================

My model reports "cuda runtime error(2): out of memory"
-------------------------------------------------------

As the error message suggests, you have run out of memory on your
GPU.  Since we often deal with large amounts of data in PyTorch,
small mistakes can rapidly cause your program to use up all of your
GPU; fortunately, the fixes in these cases are often simple.
Here are a few common things to check:

**Don't accumulate history across your training loop.**
By default, computations involving variables that require gradients
will keep history.  This means that you should avoid using such
variables in computations which will live beyond your training loops,
e.g., when tracking statistics. Instead, you should detach the variable
or access its underlying data.

Sometimes, it can be non-obvious when differentiable variables can
occur.  Consider the following training loop (abridged from `source
<https://discuss.pytorch.org/t/high-memory-usage-while-training/162>`_):

.. code-block:: python

    total_loss = 0
    for i in range(10000):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output)
        loss.backward()
        optimizer.step()
        total_loss += loss

Here, ``total_loss`` is accumulating history across your training loop, since
``loss`` is a differentiable variable with autograd history. You can fix this by
writing `total_loss += float(loss)` instead.

Other instances of this problem:
`1 <https://discuss.pytorch.org/t/resolved-gpu-out-of-memory-error-with-batch-size-1/3719>`_.

**Don't hold onto tensors and variables you don't need.**
If you assign a Tensor or Variable to a local, Python will not
deallocate until the local goes out of scope.  You can free
this reference by using ``del x``.  Similarly, if you assign
a Tensor or Variable to a member variable of an object, it will
not deallocate until the object goes out of scope.  You will
get the best memory usage if you don't hold onto temporaries
you don't need.

The scopes of locals can be larger than you expect.  For example:

.. code-block:: python

    for i in range(5):
        intermediate = f(input[i])
        result += g(intermediate)
    output = h(result)
    return output

Here, ``intermediate`` remains live even while ``h`` is executing,
because its scope extrudes past the end of the loop.  To free it
earlier, you should ``del intermediate`` when you are done with it.

**Avoid running RNNs on sequences that are too large.**
The amount of memory required to backpropagate through an RNN scales
linearly with the length of the RNN input; thus, you will run out of memory
if you try to feed an RNN a sequence that is too long.

The technical term for this phenomenon is `backpropagation through time
<https://en.wikipedia.org/wiki/Backpropagation_through_time>`_,
and there are plenty of references for how to implement truncated
BPTT, including in the `word language model <https://github.com/pytorch/examples/tree/master/word_language_model>`_ example; truncation is handled by the
``repackage`` function as described in
`this forum post <https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226>`_.

**Don't use linear layers that are too large.**
A linear layer ``nn.Linear(m, n)`` uses :math:`O(nm)` memory: that is to say,
the memory requirements of the weights
scales quadratically with the number of features.  It is very easy
to `blow through your memory <https://github.com/pytorch/pytorch/issues/958>`_
this way (and remember that you will need at least twice the size of the
weights, since you also need to store the gradients.)

**Consider checkpointing.**
You can trade-off memory for compute by using `checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_.

My GPU memory isn't freed properly
----------------------------------
PyTorch uses a caching memory allocator to speed up memory allocations. As a
result, the values shown in ``nvidia-smi`` usually don't reflect the true
memory usage. See :ref:`cuda-memory-management` for more details about GPU
memory management.
My recurrent network doesn't work with TorchScript
-------------------------------------------------------
When using the ``pack sequence -> recurrent network -> unpack sequence`` pattern in a :class:`~torch.nn.Module` with TorchScript, you may encounter runtime errors that do not occur during eager execution. This can happen due to differences in how TorchScript handles certain operations, such as indexing and CUDA operations. If you experience issues, consider the following workarounds:

1. **Avoid CUDA Operations**: If possible, perform operations on the CPU instead of CUDA, as some users have reported that this resolves certain runtime errors.

2. **Ensure Proper Indexing**: Check that all indexing operations are valid and that indices are within the expected range. TorchScript may enforce stricter checks compared to eager execution.

3. **Use `total_length` Argument**: As with data parallelism, ensure that the :attr:`total_length` argument of :func:`~torch.nn.utils.rnn.pad_packed_sequence` is used to maintain consistent sequence lengths across different parts of the model.

4. **Test with Eager Execution**: Before converting to TorchScript, thoroughly test your model in eager execution to ensure that all operations are functioning as expected.

If these steps do not resolve the issue, consider reaching out to the PyTorch community for further assistance or checking for updates that may address the problem.
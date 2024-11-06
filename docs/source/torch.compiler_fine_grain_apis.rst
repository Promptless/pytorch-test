```
   "``torch.compiler.is_compiling``", "Indicates whether a graph is executed/traced as part of torch.compile() or torch.export().", "Useful for checking if the current execution is within a compiled context."
   "``torch.compiler.is_dynamo_compiling``", "Indicates whether a graph is traced via TorchDynamo. It's stricter than torch.compiler.is_compiling() flag, as it would only be set to True when TorchDynamo is used.", "Use this to specifically check for TorchDynamo tracing."
```


``torch._dynamo.disallow_in_graph`` disallows an operator but not the function
to be present in the TorchDynamo extracted graph. Note that this is suitable
for operators and not general functions as in the case of ``_dynamo.disable``.

Let's imagine you compile your model with PyTorch. TorchDynamo is able to
extract a graph, but then you see the downstream compiler failing. For example,
the meta kernel is missing, or some Autograd dispatch key is set incorrectly
for a particular operator. Then you can mark that operator as
``disallow_in_graph``, and TorchDynamo will cause a graph break and run that
operator by using the PyTorch eager mode.

The catch is that you will have to find the corresponding Dynamo level operator,
and not the ATen level operator. See more in the Limitations section of the doc.

.. warning::
   ``torch._dynamo.disallow_in_graph`` is a global flag. If you are comparing
   different backend compilers, you might have to call ``allow_in_graph`` for
   the disallowed operator when switching to the other compiler.

``torch.compiler.allow_in_graph``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch.compiler.allow_in_graph`` is useful when the relevant function frame
has some known hard-to-support TorchDynamo feature, such as hooks and
``autograd.Function``, and you are confident that downstream PyTorch components
such as AOTAutograd can safely trace through the decorated function. When a
function is decorated with ``allow_in_graph``, TorchDynamo treats it as a
black-box and puts it as is in the generated graph.

.. warning::
   ``allow_in_graph`` skips TorchDynamo completely on the decorated function
   omitting all TorchDynamo safety checks, including graph breaks, handling
   closures, and others. Use `allow_in_graph` with caution. PyTorch downstream
   components, such as AOTAutograd rely on TorchDynamo to handle complex Python
   features, but ``allow_in_graph`` bypasses TorchDynamo. Using ``allow_in_graph``
   could lead to soundness and hard-to-debug issues.

Limitations
~~~~~~~~~~~

All the existing APIs are applied at the TorchDynamo level. Therefore, these
APIs have visibility to only what TorchDynamo sees. This can lead to confusing
scenarios.

For example, ``torch._dynamo.disallow_in_graph`` will not work for ATen operators
because they are visible to AOT Autograd. For example,
``torch._dynamo.disallow_in_graph(torch.ops.aten.add)`` will not work in the
above example.

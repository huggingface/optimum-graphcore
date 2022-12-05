"""
# Managing IPU resources from notebooks

The execution model of IPUs and notebooks means that as you experiment with different models
you might keep hold of hardware in an idle state preventing other users from using it or
your experiments might fail because you have insufficient hardware.
Releasing hardware is particularly important in notebooks as the long life time of the
underlying `ipython` kernel can keep a lock on IPUs long after you are done interacting
with the hardware.

The Graphcore frameworks operate a computational architecture of 1 model = 1 IPU device;
this means that each model will attach to specific IPUs and will only release them when
that model goes out of scope or when resources are explicitly released.
In this notebook you will learn:

- to monitor how many IPUs your notebook is currently using
- to release IPUs by detaching a model
- to reattach a model to IPUs, to continue using a model after a period of inactivity.

For more information on the basics of IPU computational architecture you may want to read
the [IPU programmer's guide]().
"""
"""
## Setup

placeholder
"""
%pip install "optimum-graphcore>=0.4, <0.5"

import os

"""
## Monitoring resources

Grapchore provides the `gc-monitor` utility for inspecting the number of available IPUs and their usage:
"""
!gc - monitor

"""
In a notebook we can run this bash command using `!` in a regular code cell. It provides detailed information on the IPUs that exist in the current partition.
The first section of the output is the `card-info`, this is generic information about the IP addresses and serial numbers of all the cards visible to the process.
The second section of the output indicates usage information of the IPU: it will indicate the user, host and PID which are attached to the different IPUs.

When monitoring IPUs it can be useful to run `gc-monitor` without displaying the static IPU information:
"""
!gc - monitor - -no - card - info

"""
Finally we can write a command that will monitor only the IPUs which are attached from this specific notebook. We do that by only displaying the IPUs attached to a specific PID:
"""
!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
Since we've not attached to any IPUs yet, there is not output.

Beyond `gc-monitor` Graphcore also provides a library for monitoring usage called `gcipuinfo` which can be used in Python, this library is not covered in this tutorials but [examples are available in the documentation]().
"""
"""
### Creating models

Now let's create some models and attach them to IPUs. The simplest way to create a small model is using the inference `pipeline` provided by the `optimum-graphcore` library.
"""
from optimum.graphcore import pipelines
sentiment_pipeline = pipelines.pipeline("sentiment-analysis")
sentiment_pipeline(["IPUs are great!", "Notebooks are easy to program in"])

"""
Now let's check how many IPUs are in use:
"""
!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
These IPUs will be associated with the model in the pipeline until:

- The `sentiment_pipeline` object goes out of scope or
- The model is explicitly detached from the IPU.

By remaining attached the model can be very fast, providing fast responses to new prompts:
"""
% % timeit
sentiment_pipeline(["IPUs are fast once the pipeline is attached", "and Notebooks are easy to program in"])

"""
If you are testing different models you might have multiple pipelines using IPUs:
"""
sentiment_pipeline_2 = pipelines.pipeline("text-classification")
sentiment_pipeline_2(["IPUs are great!", "Notebooks are easy to program in"])

"""
Checking the IPU usage we can see that we are now using 4 IPUs:
"""
!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
## Managing resources

From this we see that we are using 4 IPUs, two per active pipeline. While it may make sense for us to keep both pipelines active if we are testing both at the same time we may need to free up resources to continue experimenting with more models.

To do that we can call the `detachFromDevice` method on the model:
"""
sentiment_pipeline.model.detachFromDevice()

!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
This method has freed up the IPU resources while keeping the pipeline object available, meaning that we can quickly reattach the same pipeline to an IPU simply by calling it:
"""
% % time
sentiment_pipeline(simple_test_data)

!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
the first call is slow as the model is loaded onto the accelerator, but subsequent calls will be fast:
"""
% % time
sentiment_pipeline(simple_test_data)

"""
The other way to release resources is to let the `sentiment_pipeline` Python variable go out of scope.
There are two main ways to do that:

1. if you want to use the resources for another pipeline you can assign another variable to the same name:
"""
sentiment_pipeline = sentiment_pipeline_2

!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
2. Explicitly use `del` to delete the variables
"""
# Note that after the assignment sentiment_pipeline and sentiment_pipeline_2
# refer to the same object so both symbols must be deleted to release the resources
del sentiment_pipeline
del sentiment_pipeline_2

!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
As expected no IPUs are used by the process anymore.

Alternatively all IPUs will be released when the notebook kernel is restarted. This can be done from the Notebook graphical user interface by clicking on `Kernel > Restart`:

![Restart ipykernel](images/restart_kernel.png)

"""
"""
## Conclusion

In this simple tutorial we saw how to manage IPU resources from a notebook to make sure that we do not try to use more IPUs than are available on a single system.

For more information on using IPUs and the Poplar SDK through Jupyter notebooks please see the our [dedicated guide]().
"""

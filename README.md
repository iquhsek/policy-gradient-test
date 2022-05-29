# Policy gradient in Keras API

Another day learning with Phil. I referred to his [video](https://www.youtube.com/watch?v=IS0V8z8HXrM).

But Phil's original code is not working that well in the current keras environment. For example, to run the code, we should disable TF eager execution mode when we use custom loss function. Also, prevent keras from running experimental tensorflow functions when compiling models by

`model.compile(..., experimental_run_tf_function=False)`

## Reference

1. [Machine learning with Phil, policy gradient in keras](https://www.youtube.com/watch?v=IS0V8z8HXrM)

2. [Cannot convert a symbolic Keras input/output to a numpy array TypeError when using sampled_softmax in tensorflow 2.4](https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi)

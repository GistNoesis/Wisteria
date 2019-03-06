Here, I'll explain the design choices, regarding the speech to text, i.e. (voicetranscriptor.py)

We use mozilla commons voice dataset.

One of our main constraint is the doing the processing real-time.
This mean that we need to use a ctc loss (or provide alignment), so here we choose to do ctc loss in the deep speech style, but using a transformer architecture in place of the convolutions.

If you are just trying to just do audio to text without this real time constraint, you will probably get better results using a standard encoder-decoder architecture with a transformer.

We are using a small causal window, (i.e local attention), so that the computation remains fast later in javascript.
The neural network will not be exactly real-time, but will learn to to manage the filtering/smoothing using the available causal window.
If you give a bigger causal window, it will wait a little more (doing a little smoothing) before outputting a result, but the letter will incorporate more of the language model.

We are currently using a single-stage, but we may add a second-stage with language model, trained on text only to improve performance.

We are not using mic frequency distortion because the dataset is coming from a wide variety of microphone, though we add some noise (the same way as in musictranscriptor)

We use horovod to parallelize accross different gpus.

Using 2, 1080 ti Gpus, training time is about 2 days, but first results appear after ~ an hour.

Improving performance is an iterative process, first tweak the model parameters to make it overfit on the training set, then add some regularization terms, to make it generalize well.

If you want to contribute there is the donate button on our Readme.md 

To run : adapt horovodvoice.sh to your cluster configuration then call the script.

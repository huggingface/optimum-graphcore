"""
# Sentiment analysis (using IPUs)

Integration of the Graphcore Intelligence processing unit (IPU) and the HuggingFace transformer library means that it only takes a few lines of code to perform complex tasks which require deep learning.

In this notebook we perform **sentiment analysis**: we use natural language processing models to classify text prompts. 
We follow [this blog post by Federico Pascual](https://huggingface.co/blog/sentiment-analysis-python) and test 5 different models available on HuggingFace Hub to highlight different model properties of the models that can be leveraged for downstream tasks.

The ease of use of the `pipeline` interface lets us quickly experiment with the pre-trained models and identify which one will work best.
This simple interface means that it is extremely easy to access the fast inference performance of the IPU on your application.

![Widget inference on a text classification task](images/text_classification.png)

While this notebook is focused on using the model (inference), our [text_classification](text_classification.ipynb) notebook will show you how to fine tune a model for a specific task using the `datasets` package.
"""
"""
First of all, lets make sure your environment has the latest version of [🤗 Optimum Graphcore](https://github.com/huggingface/optimum-graphcore) available.
"""
%pip install "optimum-graphcore>=0.4, <0.5"
%pip install emoji == 0.6.0

%load_ext autoreload
%autoreload 2

import os
os.environ["POPTORCH_LOG_LEVEL"] = "ERR"

"""
## Using transformers pipelines on the IPU

The simplest way to use a model on the IPU is to use the `pipeline` function. It provides a set of models which have been validated to work on a given task, to get started choose the task and call the `pipeline` function:
"""
from optimum.graphcore import pipelines
sentiment_pipeline = pipelines.pipeline("sentiment-analysis")

"""
Because no model was supplied the pipeline defaulted to `distilbert-base-uncased-finetuned-sst-2-english`. We are helpfully warned that we should provide the `max_length` argument if we were to put this model in production, but while we are experimenting we will leave it as is.

Now it's time to test our first prompts. Let's start with some very easy to classify text:
"""
simple_test_data = ["I love you", "I hate you"]
sentiment_pipeline(simple_test_data)

"""
Reassuringly, the model got it right! And with a high degree of confidence, more than 99.9% in both cases.

You'll have noticed that this was a bit slow, it took several seconds to provide the answer. This is because of the computing architecture of the IPU, during the first execution of a pipeline, the model needs to be compiled and loaded into the IPU. While this takes some time, these overheads only the first execution of a pipeline! 

On subsequent prompts it is much faster:
"""
sentiment_pipeline(simple_test_data)

"""
Now that was much faster! We can use the `%%timeit` cell magic to check how fast:
"""
% % timeit
sentiment_pipeline(simple_test_data)

"""
It takes on the order of ~1ms per prompt, this is fast!
"""
"""
### Customising the IPU resources

Depending on your system you may have 4, 16 or 64 IPUs available. IPUs are designed from the ground up to make it easy to scale applications to large numbers of processors working together, however, in this case it is not needed, 1 IPU is sufficient.

We're going to make sure that we are using a single IPU so that other users or other applications that we are running in the background are not affected. To do that, we define an `inference_config` dictionary which contains arguments that will be passed to the `optimum.graphcore.IPUConfig` object which controls the accelerator. 

We recreate our pipeline with the new settings:

"""
inference_config = dict(layers_per_ipu=[40], ipus_per_replica=1, enable_half_partials=True, matmul_proportion=0.6)
sentiment_pipeline = pipelines.pipeline("sentiment-analysis", ipu_config_kwargs=inference_config)
data = ["I love you", "I hate you"]
sentiment_pipeline(simple_test_data)

"""
It still works as expected.

"""
"""
### Asking more complex questions

Now, our initial prompts are trivial to classify, what if we asked the pipeline to classify a more ambiguous sentences?
"""
ambiguous_prompts = [
    "How are you today?",
    "I'm a little tired, I didn't sleep well, but I hope it gets better"
]

"""
The first sentence is perfectly neutral, while the second is a mix of negative and positive sentiment.
A good answer from the model would reflect the ambiguous nature of the prompt.
"""
sentiment_pipeline(ambiguous_prompts)

"""
The model only supports two labels: "POSITIVE" and "NEGATIVE"; neither of which really captures the sentiment of those messages.
While we see a slight drop in the confidence of the model, but it does not feel sufficient to reflect message.

The imprecise classification of these prompts would affect any downstream task: if we were trying to derive some insights from the model on customer satisfaction we may have an overly optimistic view of performance and derive the wrong conclusions.

To resolve this issue we need to try more models, which accommodate finer grained classification.
"""
"""
## Trying more models

[The blog post]() we are following suggests a number of other models, lets try them all to see if they perform better on our ambiguous prompts!

The first one is `finiteautomata/bertweet-base-sentiment-analysis` a [RoBERTa model trained on 40 thousand tweets](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) collected before 2018. Using it is as simple as giving the name of the 🤗 hub repository as the model argument:
"""
tweet_model = pipelines.pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis", ipu_config_kwargs=inference_config,
)
print(simple_test_data)
tweet_model(simple_test_data)

"""
Unsurprisingly, the model correctly classifies our simple prompt. Now let's try our more ambiguous prompt. 
"""
tweet_model(["How are you today?", "I'm a little tired, I didn't sleep well, but I hope it gets better"])

"""
That is much better: the model classifies the first prompt as neutral and the second as negative. The addition of a "NEU" (neutral) class gives the model the flexibility to correctly identify statements which do not fit as positive or negative.

The challenge of the second prompt is that it has multiple clauses which capture different sentiments. To get a better result on it you might separate it out into multiple prompts that are better suited to the model. For example we can split on `,` to classify each clause of the sentence on its own:
"""
out = tweet_model([
    "How are you today?",
    *"I'm a little tired, I didn't sleep well, but I hope it gets better".split(","),
])
out

"""
Here each parts of the sentence is correctly classified, how you choose to process those sentence parts will depend on what you need to use the results of the sentiment analysis for.

We can do small changes to the prompt to get an intuition of how the model responds to changes in grammar. Below the last part of the ambiguous sentence is changed to be more optimistic:
"""
print(f"Previous score: {out[-1]}")

tweet_model(["but it is getting better"])

"""
As a consequence of that change the score associated with the positive label has gone up, matching the desired behavior of the model.
"""
"""
### A model finetuned on tweets

The next model discussed in the blog post is the [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model, it is a RoBERTa-Base which was trained on 124M tweets collected between 2018 and 2021. This data makes the model much more recent than the previous pre-trained checkpoint.

As before this model is trivial to load through the pipeline API:
"""
from pprint import pprint
newer_model = pipelines.pipeline(
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", ipu_config_kwargs=inference_config
)
out = newer_model(simple_test_data + ambiguous_prompts)
# print prompts and predicted labels side by side
pprint([*zip(simple_test_data + ambiguous_prompts, out)])

"""
This model performs similarly on these prompts as the model explored in the previous section. Differences in the model may not be apparent until we start prompting about recent events.

If we ask the previous model to classify a statement about the Coronavirus pandemic we get different results between the models:
"""
coronavirus_prompt = ["Coronavirus is increasing"]
old = tweet_model(coronavirus_prompt)
new = newer_model(coronavirus_prompt)
print(f"Older model score: {old}")
print(f"Newer model score: {new}")

"""
The newer model has a strong negative connotation for Coronavirus while the older model sees it as a neutral statement. This simple experiment shows the importance of testing and fine-tuning models regularly to make sure that sentiment analysis continues to be accurate as connotations of certain words evolve.
"""
# We stop using these pipelines in the rest of the notebook so we detach from the
# IPU devices to allow us to test additional models for more details see the section
# on managing resources at the end of the document.
tweet_model.model.detachFromDevice()
newer_model.model.detachFromDevice()

"""
### Multi-lingual model

The next model has an interesting feature: it is multi-lingual. It was trained on a dataset of English, Dutch, German, Spanish, Italian and French text, it can be prompted in any of these languages and should correctly classify the text inputs.

The model is `nlptown/bert-base-multilingual-uncased-sentiment` a BERT checkpoint fine-tuned on ~700k reviews in 6 languages and predicts between 1 and 5 stars for each prompt.
"""
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
multilingual_model = pipelines.pipeline(
    model=model_name, ipu_config_kwargs=inference_config
)
print(simple_test_data)
multilingual_model(simple_test_data)

"""
It successfully classifies our simple input, now let's see how it fares with our ambiguous input:
"""
multilingual_model([
    "How are you today?",
    "I'm a little tired, I didn't sleep well, but I hope it gets better"
])

"""
While it is a bit optimistic about our first prompt, it's guess is given with a fairly low confidence, and it identifies the second prompt as neutral with a median score of 3.

Now let's translate our prompts and ask it the same questions in French:
"""
ambiguous_in_french = [
    "Comment vas-tu aujourd'hui?",
    "Je suis un peu fatigue, je n'ai pas bien dormi mais j'espere que la journee s'ameliore",
]
multilingual_model(ambiguous_in_french)


"""
The model gives very similar results in both languages!

Now let us revisit our first model, could it also work with this multi-lingual input?
"""
print(sentiment_pipeline(ambiguous_prompts))
print(sentiment_pipeline(ambiguous_in_french))

"""
Unsurprisingly it cannot, its prediction are not stable across the two languages. However this model did not perform very well on the ambiguous prompts. If we re-use some of the more precise English-only models and run them on more obvious prompts: 
"""
simple_french_input = [
    "Ce film est excellent",  # This film is excellent
    "Le produit ne marche pas du tout"  # The tool does not work at all
]
print(newer_model(simple_french_input))
print(multilingual_model(simple_french_input))

"""
In this case we see the multi-lingual model correctly predict strongly positive and negative labels, while the other model predicts a positive message (correct) and a neutral (expected LABEL_0).
"""
"""
### Other models

Models can be finetuned to extract different classes from text. The `bhadresh-savani/distilbert-base-uncased-emotion` checkpoint is a DistilBERT checkpoint tuned to identify the emotion associated with a prompt:
"""
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_model = pipelines.pipeline(model=model_name, ipu_config_kwargs=inference_config)
emotion_model(simple_test_data)

"""
This model can be prompted with sentences which include different emotions:
"""
emotion_model([
    "How are you today?",
    "Don't make me go out, it's too cold!",
    "What is happening, I don't understand",
    "Where did you come from?",
])


"""
## Managing resources

You have now created multiple models. The IPU architecture means that each pipeline will be attached to it's own IPUs.

Grapchore provides the `gc-monitor` utility for inspecting the number of available IPUs and their usage:
"""
!gc - monitor

"""
Now it can also be useful to capture a summary of the IPU usage of the current process:
"""
!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
from this we see that we are using 4 IPUs, one per active pipeline. To continue experimenting with more models we might need to free up some devices, to do that we can call the `detachFromDevice` method on the model:
"""
emotion_model.model.detachFromDevice()

"""
This method will free up the IPU resources while keeping the pipeline object available, meaning that we can quickly reattach the same pipeline to an IPU simply by calling it:
"""
% % time
emotion_model(simple_test_data)

"""
the first call is slow as the model is loaded onto the accelerator, but subsequent calls will be fast:
"""
% % time
emotion_model(simple_test_data)

"""
Let's detach most pipelines:
"""
sentiment_pipeline.model.detachFromDevice()
emotion_model.model.detachFromDevice()

!gc - monitor - -no - card - info | grep ${os.getpid()}

"""
## Using a larger model

The optimum library supports several sizes of models for many of the standard architectures, in this section we load a checkpoint which uses roBERTa large to perform the same task.

Larger models will take longer to execute but may provide better predictions in a broader range of cases. As an example we load the [`j-hartmann/sentiment-roberta-large-english-3-classes`](https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes) checkpoint which uses the roBERTa-Large model trained on 5000 manually annotated social media posts. 
"""
larger_model = pipelines.pipeline(
    "sentiment-analysis",
    "j-hartmann/sentiment-roberta-large-english-3-classes",
    ipu_config_kwargs=inference_config
)
larger_model(simple_test_data)

"""
As before the it succeeds on our simple example. We can check the latency of that model:
"""
% % timeit
larger_model(simple_test_data)

"""
The model takes about 4ms to execute per prompt, as expected this is slower than the earlier pipelines which used a smaller model.
"""
larger_model(["How are you today?", "I'm a little tired, I didn't sleep well, but I hope it gets better"])

"""
## Next steps

This notebook has followed the steps of a blog post, showing how easy using IPUs for sentiment analysis is.
The integration with the pipeline interface makes thousands of models available on the HuggingFace hub easy to use on the IPU, while following the blog post we experimented with 6 different checkpoints testing the properties of the different models.

There are [hundreds more available on the Hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment), try them out, we think they'll work but, if you hit an error, [raise an issue or open a pull request](https://github.com/huggingface/optimum-graphcore/issues) and we'll do our best to fix it! 🤗
If you have your own dataset, the [text classfication notebook](text_classification.ipynb) will show you how a simple way to fine-tune a classification model tailored to your use case.
"""
"""
You may also want to check out one of the other tasks available through the `pipeline` API:
"""
pipelines.list_tasks()

"""

- Explore another approach to sentiment analysis through text-generation
"""

"""
# Sentiment analysis (using IPUs)

Integration of the Graphcore Intelligence Processing Unit (IPU) and the Hugging Face transformer library means that it only takes a few lines of code to perform complex tasks which require deep learning.

In this notebook we perform **sentiment analysis**: we use natural language processing models to classify text prompts. 
We follow [this blog post by Federico Pascual](https://huggingface.co/blog/sentiment-analysis-python) and test 5 different models available on Hugging Face Hub to highlight different model properties of the models that can be leveraged for downstream tasks.

The ease of use of the `pipeline` interface lets us quickly experiment with the pre-trained models and identify which one will work best.
This simple interface means that it is extremely easy to access the fast inference performance of the IPU on your application.

<img src="images/text_classification.png" alt="Widget inference on a text classification task" style="width:500px;">


While this notebook is focused on using the model (inference), our [text_classification](text_classification.ipynb) notebook will show you how to fine tune a model for a specific task using the [`datasets`](https://huggingface.co/docs/datasets/index) package.
"""
"""
First of all, let's make sure your environment has the latest version of [🤗 Optimum Graphcore](https://github.com/huggingface/optimum-graphcore) available.
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
The pipeline defaults to `distilbert-base-uncased-finetuned-sst-2-english`, a checkpoint managed by Hugging Face because we did not provide a specific model.
We are helpfully warned that we should explicitly specify a maximum sequence length through the `max_length` argument if we were to put this model in production, but while we experiment we will leave it as is.

Now it's time to test our first prompts. Let's start with some very easy to classify text:
"""
simple_test_data = ["I love you", "I hate you"]
sentiment_pipeline(simple_test_data)

"""
Reassuringly, the model got it right! And with a high degree of confidence, more than 99.9% in both cases.

The first call to the pipeline was a bit slow, it took several seconds to provide the answer. This behaviour is due to compilation of the model which happens on the first call.
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
### Other tasks supported by IPU pipelines

This simple interface provides access to a number of other tasks which can be listed through the `list_tasks` function:
"""
pipelines.list_tasks()

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

[The blog post](https://huggingface.co/blog/sentiment-analysis-python) we are following suggests a number of other models, lets try them all to see if they perform better on our ambiguous prompts!

The first one is `finiteautomata/bertweet-base-sentiment-analysis` a [RoBERTa model trained on 40 thousand tweets](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) collected before 2018. Using it is as simple as giving the name of the 🤗 hub repository as the model argument:
"""
tweet18_pipeline = pipelines.pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis", ipu_config_kwargs=inference_config,
)
print(simple_test_data)
tweet18_pipeline(simple_test_data)

"""
Unsurprisingly, the model correctly classifies our simple prompt. Now let's try our more ambiguous prompt. 
"""
tweet18_pipeline(["How are you today?", "I'm a little tired, I didn't sleep well, but I hope it gets better"])

"""
That is much better: the model classifies the first prompt as neutral and the second as negative. The addition of a "NEU" (neutral) class gives the model the flexibility to correctly identify statements which do not fit as positive or negative.

The challenge of the second prompt is that it has multiple clauses which capture different sentiments. To get a better result on it you might separate it out into multiple prompts that are better suited to the model. For example we can split on `,` to classify each clause of the sentence on its own:
"""
split_prompts_answer = tweet18_pipeline([
    "How are you today?",
    *"I'm a little tired, I didn't sleep well, but I hope it gets better".split(","),
])
split_prompts_answer

"""
Here each parts of the sentence is correctly classified, how you choose to process those sentence parts will depend on what you need to use the results of the sentiment analysis for.

We can do small changes to the prompt to get an intuition of how the model responds to changes in grammar. Below the last part of the ambiguous sentence is changed to be more optimistic:
"""
print(f"Previous score: {split_prompts_answer[-1]}")

tweet18_pipeline(["but it is getting better"])

"""
As a consequence of that change the score associated with the positive label has gone up, matching the desired behavior of the model.
"""
"""
### A model finetuned on tweets

The next model discussed in the blog post is the [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model, it is a RoBERTa-Base which was trained on 124M tweets collected between 2018 and 2021. This data makes the model much more recent than the previous pre-trained checkpoint.

As before this model is trivial to load through the pipeline API:
"""
from pprint import pprint
tweet21_pipeline = pipelines.pipeline(
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", ipu_config_kwargs=inference_config
)
out = tweet21_pipeline(simple_test_data + ambiguous_prompts)
# print prompts and predicted labels side by side
pprint([*zip(simple_test_data + ambiguous_prompts, out)])

"""
This model performs similarly on these prompts as the model explored in the previous section. Differences in the model may not be apparent until we start prompting about recent events.

If we ask the previous model to classify a statement about the Coronavirus pandemic we get different results between the models:
"""
coronavirus_prompt = ["Coronavirus is increasing"]
old = tweet18_pipeline(coronavirus_prompt)
new = tweet21_pipeline(coronavirus_prompt)
print(f"Older model score: {old}")
print(f"Newer model score: {new}")

"""
The newer model has a strong negative connotation for Coronavirus while the older model sees it as a neutral statement. This simple experiment shows the importance of testing and fine-tuning models regularly to make sure that sentiment analysis continues to be accurate as connotations of certain words evolve.
"""
"""
We are done using these pipelines in the rest of the notebook so we detach from the IPU devices:
"""
!gc - monitor - -no - card - info | grep ${os.getpid()}
tweet18_pipeline.model.detachFromDevice()
tweet21_pipeline.model.detachFromDevice()

"""
This will allow us to test additional models, for more details on managing IPU resources from a notebook you can consult our [notebook on managing IPU resources](managing_ipu_resources.ipynb).
"""
"""
### Multi-lingual model

The next model has an interesting feature: it is multi-lingual. It was trained on a dataset of English, Dutch, German, Spanish, Italian and French text, it can be prompted in any of these languages and should correctly classify the text inputs.

The model is `nlptown/bert-base-multilingual-uncased-sentiment` a BERT checkpoint fine-tuned on ~700k reviews in 6 languages, which gives a rating between 1 and 5 stars for each prompt. 1 start indicates a very negative sentiment, while 5 starts corresponds to a very positive text.
"""
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
multilingual_pipeline = pipelines.pipeline(
    model=model_name, ipu_config_kwargs=inference_config
)
print(simple_test_data)
multilingual_pipeline(simple_test_data)

"""
It successfully classifies our simple input, now let's see how it fares with our ambiguous input:
"""
multilingual_pipeline([
    "How are you today?",  # How are you today?
    "I'm a little tired, I didn't sleep well, but I hope it gets better"
    # "I'm a little tired, I didn't sleep well, but I hope it gets better
])

"""
While it is a bit optimistic about our first prompt, it's guess is given with a fairly low confidence, and it identifies the second prompt as neutral with a median score of 3.

Now let's translate our prompts and ask it the same questions in French:
"""
ambiguous_in_french = [
    "Comment vas-tu aujourd'hui?",
    "Je suis un peu fatigue, je n'ai pas bien dormi mais j'espere que la journee s'ameliore",
]
multilingual_pipeline(ambiguous_in_french)


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
print(tweet21_pipeline(simple_french_input))
print(multilingual_pipeline(simple_french_input))

"""
In this case we see the multi-lingual model correctly predict strongly positive and negative labels, while the other model predicts a positive message (correct) and a neutral (expected LABEL_0).
"""
"""
### Other models

Models can be finetuned to extract different classes from text. The `bhadresh-savani/distilbert-base-uncased-emotion` checkpoint is a DistilBERT checkpoint tuned to identify the emotion associated with a prompt:
"""
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
emotion_pipeline = pipelines.pipeline(model=model_name, ipu_config_kwargs=inference_config)
emotion_pipeline(simple_test_data)

"""
This model can be prompted with sentences which include different emotions:
"""
emotion_pipeline([
    "How are you today?",
    "Don't make me go out, it's too cold!",
    "What is happening, I don't understand",
    "Where did you come from?",
])


"""
We detach from the remaining pipelines.
"""
sentiment_pipeline.model.detachFromDevice()
emotion_pipeline.model.detachFromDevice()

"""
## Using a larger model

The optimum library supports several sizes of models for many of the standard architectures, in this section we load a checkpoint which uses roBERTa large to perform the same task.

Larger models will take longer to execute but may provide better predictions in a broader range of cases. As an example we load the [`j-hartmann/sentiment-roberta-large-english-3-classes`](https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes) checkpoint which uses the roBERTa-Large model trained on 5000 manually annotated social media posts. 
"""
larger_pipeline = pipelines.pipeline(
    "sentiment-analysis",
    "j-hartmann/sentiment-roberta-large-english-3-classes",
    ipu_config_kwargs=inference_config
)
larger_pipeline(simple_test_data)

"""
As before the it succeeds on our simple example. We can check the latency of that model:
"""
% % timeit
larger_pipeline(simple_test_data)

"""
The model takes about 4ms to execute per prompt, as expected this is slower than the earlier pipelines which used a smaller model.
"""
larger_pipeline(["How are you today?", "I'm a little tired, I didn't sleep well, but I hope it gets better"])

"""
## Next steps

This notebook has followed the steps of a blog post, showing how easy using IPUs for sentiment analysis is.
The integration with the pipeline interface makes thousands of models available on the Hugging Face hub easy to use on the IPU, while following the blog post we experimented with 6 different checkpoints testing the properties of the different models.

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

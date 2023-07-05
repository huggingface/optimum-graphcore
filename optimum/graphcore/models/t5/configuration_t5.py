from transformers import T5Config


# Config class for testing
# Used to models that don't have a mapping in upstream transformers
class T5EncoderConfig(T5Config):
    pass

from traitlets.config import Config

from src.constants import REMOVE_OUTPUT_TAG


def configure_tag_removal_preprocessor():
    c = Config()
    c.TagRemovePreprocessor.remove_all_outputs_tags = (REMOVE_OUTPUT_TAG, )
    c.TagRemovePreprocessor.enabled = True
    return c

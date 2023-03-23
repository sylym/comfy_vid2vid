import boto3
from io import BytesIO
import copy
import random
import os
from functools import lru_cache
from typing import Callable, Dict, Optional, Union, List
from urllib.parse import urlparse

import torch

from transformers import PreTrainedModel, PreTrainedTokenizer, CLIPTokenizer
from diffusers.utils import logging


from diffusers.utils.import_utils import ENV_VARS_TRUE_VALUES
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "").upper() in ENV_VARS_TRUE_VALUES


logger = logging.get_logger(__name__)


class TextualInversionLoaderMixin:
    r"""
    Mixin class for adding textual inversion tokens and embeddings to the tokenizer and text encoder with method:
    - [`~TextualInversionLoaderMixin.load_textual_inversion_embeddings`]
    - [`~TextualInversionLoaderMixin.add_textual_inversion_embedding`]
    """

    def load_textual_inversion_embeddings(
        self,
        embedding_path_dict_or_list: Union[Dict[str, str], List[Dict[str, str]]],
        allow_replacement: bool = False,
        boto3_session: Optional["boto3.Session"] = None,
    ):
        r"""
        Loads textual inversion embeddings and adds them to the tokenizer's vocabulary and the text encoder's embeddings.
        Arguments:
            embeddings_path_dict_or_list (`Dict[str, str]` or `List[str]`):
                Dictionary of token to embedding path or List of embedding paths to embedding dictionaries.
                The dictionary must have the following keys:
                    - `token`: name of the token to be added to the tokenizers' vocabulary
                    - `embedding`: path to the embedding of the token to be added to the text encoder's embedding matrix
                The list must contain paths to embedding dictionaries where the keys are the tokens and the
                values are the embeddings (same as above dictionary definition).
            allow_replacement (`bool`, *optional*, defaults to `False`):
                Whether to allow replacement of existing tokens in the tokenizer's vocabulary. If `False`
                and a token is already in the vocabulary, an error will be raised.
            boto3_session (`boto3.Session`, *optional*):
                Boto3 session to use to load the embeddings from S3. If not provided and loading from s3, will use the default boto3 credential.
                See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html for more details.
        Returns:
            None
        """
        # Validate that inheriting class instance contains required attributes
        self._validate_method_call(self.load_textual_inversion_embeddings)

        self.boto3_session = boto3_session

        if isinstance(embedding_path_dict_or_list, dict):
            for token, embedding_path in embedding_path_dict_or_list.items():
                if embedding_path.startswith("s3://"):
                    embedding_dict = self._load_from_s3(embedding_path)
                else:
                    embedding_dict = torch.load(embedding_path, map_location=self.text_encoder.device)
                embedding, is_multi_vec_token = self._extract_embedding_from_dict(embedding_dict)

                self._validate_token_update(token, allow_replacement, is_multi_vec_token)
                self.add_textual_inversion_embedding(token, embedding)
        elif isinstance(embedding_path_dict_or_list, list):
            for embedding_path in embedding_path_dict_or_list:
                if embedding_path.startswith("s3://"):
                    embedding_dict = self._load_from_s3(embedding_path)
                else:
                    embedding_dict = torch.load(embedding_path, map_location=self.text_encoder.device)
                token = self._extract_token_from_dict(embedding_dict)
                embedding, is_multi_vec_token = self._extract_embedding_from_dict(embedding_dict)

                self._validate_token_update(token, allow_replacement, is_multi_vec_token)
                self.add_textual_inversion_embedding(token, embedding)
        else:
            raise ValueError(
                f"Type {type(embedding_path_dict_or_list)} is invalid. The value passed to `embedding_path_dict_or_list` "
                "must be a dictionary that maps a token to it's embedding file path "
                "or a list of paths to embedding files containing embedding dictionaries."
            )

    def add_textual_inversion_embedding(self, token: str, embedding: torch.Tensor):
        r"""
        Adds a token to the tokenizer's vocabulary and an embedding to the text encoder's embedding matrix.
        Arguments:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
            embedding (`torch.Tensor`):
                The embedding of the token to be added to the text encoder's embedding matrix
        Returns:
            None
        """
        # NOTE: Not clear to me that we intend for this to be a public/exposed method.
        # Validate that inheriting class instance contains required attributes
        self._validate_method_call(self.load_textual_inversion_embeddings)

        embedding = embedding.to(self.text_encoder.dtype)

        if not isinstance(self.tokenizer, MultiTokenCLIPTokenizer):
            if token in self.tokenizer.get_vocab():
                # If user has allowed replacement and the token exists, we only need to
                # extract the existing id and update the embedding
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            else:
                # If the token does not exist, we add it to the tokenizer, then resize and update the
                # text encoder acccordingly
                self.tokenizer.add_tokens([token])

                token_id = self.tokenizer.convert_tokens_to_ids(token)
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
        else:
            if token in self.tokenizer.token_map:
                # If user has allowed replacement and the token exists, we need to
                # remove all existing tokens associated with the old embbedding and
                # upddate with the new ones
                indices_to_remove = []
                for token_to_remove in self.tokenizer.token_map[token]:
                    indices_to_remove.append(self.tokenizer.get_added_vocab()[token_to_remove])

                    # Remove old  tokens from tokenizer
                    self.tokenizer.added_tokens_encoder.pop(token_to_remove)

                # Convert indices to remove to tensor
                indices_to_remove = torch.LongTensor(indices_to_remove)

                # Remove old tokens from text encoder
                token_embeds = self.text_encoder.get_input_embeddings().weight.data
                indices_to_keep = torch.arange(0, token_embeds.shape[0])
                indices_to_keep = indices_to_keep[indices_to_keep != indices_to_remove].squeeze()
                token_embeds = token_embeds[indices_to_keep]

                # Downsize text encoder
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))

                # Remove token from map so MultiTokenCLIPTokenizer doesn't complain
                # on update
                self.tokenizer.token_map.pop(token)

            # Update token with new embedding
            embedding_dims = len(embedding.shape)
            num_vec_per_token = 1 if embedding_dims == 1 else embedding.shape[0]

            self.tokenizer.add_placeholder_tokens(token, num_vec_per_token=num_vec_per_token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)

            if embedding_dims > 1:
                for i, token_id in enumerate(token_ids):
                    self.text_encoder.get_input_embeddings().weight.data[token_id] = embedding[i]
            else:
                self.text_encoder.get_input_embeddings().weight.data[token_ids] = embedding

    def _extract_embedding_from_dict(self, embedding_dict: Dict[str, str]) -> torch.Tensor:
        r"""
        Extracts the embedding from the embedding dictionary.
        Arguments:
            embedding_dict (`Dict[str, str]`):
                The embedding dictionary loaded from the embedding path
        Returns:
            embedding (`torch.Tensor`):
                The embedding to be added to the text encoder's embedding matrix
            is_multi_vec_token (`bool`):
                Whether the embedding is a multi-vector token or not
        """
        is_multi_vec_token = False
        # auto1111 embedding case
        if "string_to_param" in embedding_dict:
            embedding_dict = embedding_dict["string_to_param"]
            embedding = embedding_dict["*"]
        else:
            embedding = list(embedding_dict.values())[0]

        if len(embedding.shape) > 1:
            # If the embedding has more than one dimension,
            # We need to ensure the tokenizer is a MultiTokenTokenizer
            # because there is branching logic that depends on that class
            if not isinstance(self.tokenizer, MultiTokenCLIPTokenizer):
                raise ValueError(
                    f"{self.__class__.__name__} requires `self.tokenizer` of type `MultiTokenCLIPTokenizer` for loading embeddings with more than one dimension."
                )
            is_multi_vec_token = True

        return embedding, is_multi_vec_token

    def _extract_token_from_dict(self, embedding_dict: Dict[str, str]) -> str:
        r"""
        Extracts the token from the embedding dictionary.
        Arguments:
            embedding_dict (`Dict[str, str]`):
                The embedding dictionary loaded from the embedding path
        Returns:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
        """
        # auto1111 embedding case
        if "string_to_param" in embedding_dict:
            token = embedding_dict["name"]
            return token

        return list(embedding_dict.keys())[0]

    def _validate_method_call(self, method: Callable):
        r"""
        Validates that the method is being called from a class instance that has the required attributes.
        Arguments:
            method (`function`):
                The class's method being called
        Raises:
            ValueError:
                If the method is being called from a class instance that does not have
                the required attributes, the method will not be callable.
        Returns:
            None
        """
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PreTrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling `{method.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder, PreTrainedModel):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `PreTrainedModel` for calling `{method.__name__}`"
            )

    def _validate_token_update(self, token, allow_replacement=False, is_multi_vec_token=False):
        r"""Validates that the token is not already in the tokenizer's vocabulary.
        Arguments:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
            allow_replacement (`bool`):
                Whether to allow replacement of the token if it already exists in the tokenizer's vocabulary
            is_multi_vec_token (`bool`):
                Whether the embedding is a multi-vector token or not
        Raises:
            ValueError:
                If the token is already in the tokenizer's vocabulary and `allow_replacement` is False.
        Returns:
            None
        """
        if (not is_multi_vec_token and token in self.tokenizer.get_vocab()) or (
            is_multi_vec_token and token in self.tokenizer.token_map
        ):
            if allow_replacement:
                logger.info(
                    f"Token {token} already in tokenizer vocabulary. Overwriting existing token and embedding with the new one."
                )
            else:
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name."
                )

    @lru_cache
    def _load_from_s3(self, s3_uri):
        r"""Loads a file from the given s3 URI into working memory.
        Arguments:
            s3_uri (`str`):
                The s3 URI to load the embedding file from.
        Returns:
            `torch.Tensor`:
                The embedding to be added to the text encoder's embedding matrix.
        """
        assert s3_uri[:5] == "s3://", f"Invalid s3 URI: {s3_uri}"

        s3_client = self.boto3_session.client("s3") if self.boto3_session else boto3.client("s3")

        # Parse URI for bucket and key
        s3_bucket, s3_key = urlparse(s3_uri).netloc, urlparse(s3_uri).path.lstrip("/")

        with BytesIO() as f:
            s3_client.download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
            f.seek(0)
            return torch.load(f, map_location=self.text_encoder.device)


class MultiTokenCLIPTokenizer(CLIPTokenizer):
    """Tokenizer for CLIP models that have multi-vector tokens."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_map = {}

    def add_placeholder_tokens(self, placeholder_token, *args, num_vec_per_token=1, **kwargs):
        r"""Adds placeholder tokens to the tokenizer's vocabulary.
        Arguments:
            placeholder_token (`str`):
                The placeholder token to be added to the tokenizers' vocabulary and token map.
            num_vec_per_token (`int`):
                The number of vectors per token. Defaults to 1.
            *args:
                The arguments to be passed to the tokenizer's `add_tokens` method.
            **kwargs:
                The keyword arguments to be passed to the tokenizer's `add_tokens` method.
        Returns:
            None
        """
        output = []
        if num_vec_per_token == 1:
            self.add_tokens(placeholder_token, *args, **kwargs)
            output.append(placeholder_token)
        else:
            output = []
            for i in range(num_vec_per_token):
                ith_token = placeholder_token + f"_{i}"
                self.add_tokens(ith_token, *args, **kwargs)
                output.append(ith_token)
        # handle cases where there is a new placeholder token that contains the current placeholder token but is larger
        for token in self.token_map:
            if token in placeholder_token:
                raise ValueError(
                    f"The tokenizer already has placeholder token {token} that can get confused with"
                    f" {placeholder_token}keep placeholder tokens independent"
                )
        self.token_map[placeholder_token] = output

    def replace_placeholder_tokens_in_text(self, text, vector_shuffle=False, prop_tokens_to_load=1.0):
        r"""Replaces placeholder tokens in text with the tokens in the token map.
        Opttionally, implements:
            a) vector shuffling (https://github.com/rinongal/textual_inversion/pull/119)where
            shuffling tokens were found to force the model to learn the concepts more descriptively.
            b) proportional token loading so that not every token in the token map is loaded on each call;
            used as part of progressive token loading during training which can improve generalization
            during inference.
        Arguments:
            text (`str`):
                The text to be processed.
            vector_shuffle (`bool`):
                Whether to shuffle the vectors in the token map. Defaults to False.
            prop_tokens_to_load (`float`):
                The proportion of tokens to load from the token map. Defaults to 1.0.
        Returns:
            `str`: The processed text.
        """
        if isinstance(text, list):
            output = []
            for i in range(len(text)):
                output.append(self.replace_placeholder_tokens_in_text(text[i], vector_shuffle=vector_shuffle))
            return output
        for placeholder_token in self.token_map:
            if placeholder_token in text:
                tokens = self.token_map[placeholder_token]
                tokens = tokens[: 1 + int(len(tokens) * prop_tokens_to_load)]
                if vector_shuffle:
                    tokens = copy.copy(tokens)
                    random.shuffle(tokens)
                text = text.replace(placeholder_token, " ".join(tokens))
        return text

    def __call__(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0, **kwargs):
        """Wrapper around [`~transformers.tokenization_utils.PreTrainedTokenizerBase.__call__`] method
        but first replace placeholder tokens in text with the tokens in the token map.
        Returns:
            [`~transformers.tokenization_utils_base.BatchEncoding`]
        """
        return super().__call__(
            self.replace_placeholder_tokens_in_text(
                text,
                vector_shuffle=vector_shuffle,
                prop_tokens_to_load=prop_tokens_to_load,
            ),
            *args,
            **kwargs,
        )

    def encode(self, text, *args, vector_shuffle=False, prop_tokens_to_load=1.0, **kwargs):
        """Wrapper around the tokenizer's [`transformers.tokenization_utils.PreTrainedTokenizerBase.encode`] method
        but first replaces placeholder tokens in text with the tokens in the token map.
        Arguments:
            text (`str`):
                The text to be encoded.
            *args:
                The arguments to be passed to the tokenizer's `encode` method.
            vector_shuffle (`bool`):
                Whether to shuffle the vectors in the token map. Defaults to False.
            prop_tokens_to_load (`float`):
                The proportion of tokens to load from the token map. Defaults to 1.0.
            **kwargs:
                The keyword arguments to be passed to the tokenizer's `encode` method.
        Returns:
            List[`int`]: sequence of ids (integer)
        """
        return super().encode(
            self.replace_placeholder_tokens_in_text(
                text,
                vector_shuffle=vector_shuffle,
                prop_tokens_to_load=prop_tokens_to_load,
            ),
            *args,
            **kwargs,
        )
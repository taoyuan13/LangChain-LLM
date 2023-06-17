import logging
import os
import uuid

import torch

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    # "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec-base": "E:/AIProjects/chatglm-6b-int4",
    "text2vec-large": "GanymedeNil/text2vec-large-chinese",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "moka-ai/m3e-base",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec-base"
# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledgebase")

# 缓存知识库数量
CACHED_VS_NUM = 1
# 文本分句长度
SENTENCE_SIZE = 100

USER_NAME = uuid.uuid4().hex

logger.info(f"""
loading embedding model config...
embedding model: {EMBEDDING_MODEL}
embedding device: {EMBEDDING_DEVICE}
dir: {embedding_model_dict[EMBEDDING_MODEL]}
username: {USER_NAME}
""")

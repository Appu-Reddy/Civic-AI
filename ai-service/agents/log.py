import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

# Reduce third-party library noise
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

logger = logging.getLogger("ma-rag")

def stage(message: str):
    logger.info(message)
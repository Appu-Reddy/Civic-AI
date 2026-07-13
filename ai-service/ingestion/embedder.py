import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model(model_name=MODEL_NAME):
	return SentenceTransformer(model_name)


def generate_embeddings(chunks, model):
	if not chunks:
		return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
	return np.asarray(model.encode(list(chunks), 
							convert_to_numpy=True, 
							normalize_embeddings=True, 
							show_progress_bar=False), 
							dtype=np.float32)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("chunks", nargs="+")
	parser.add_argument("--model", default=MODEL_NAME)
	args = parser.parse_args()
	print(generate_embeddings(args.chunks, load_model(args.model)).shape)

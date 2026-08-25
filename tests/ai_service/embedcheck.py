from ai_service.ingestion.parser import parse_directory
from ai_service.ingestion.chunker import chunk_pages

PDF_DIR = "knowledge-base/PDFs"   # change if needed

for chunk in chunk_pages(parse_directory(PDF_DIR)):
    print("=" * 80)
    print(f"Chunk ID     : {chunk['chunk_id']}")
    print(f"Source       : {chunk['source_file']}")
    print(f"Page         : {chunk['page_number']}")
    print(f"Article Ref  : {chunk.get('article_ref')}")
    print(f"Tokens       : {chunk['token_count']}")
    print(chunk["text"][:500])
    print()
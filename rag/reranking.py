from langchain_openai import ChatOpenAI
from __future__ import annotations

import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import Extra

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from llm_components.chain import GeneralChain
from llm_components.prompt_templates import RerankingTemplate
from settings import settings


class LLMReranker:
    @staticmethod
    def generate_response(
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        reranking_template = RerankingTemplate()
        prompt_template = reranking_template.create_template(keep_top_k=keep_top_k)

        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID)
        chain = GeneralChain().get_chain(
            llm=model, output_key="rerank", template=prompt_template
        )

        stripped_passages = [
            stripped_item for item in passages if (stripped_item := item.strip())
        ]
        passages = reranking_template.separator.join(stripped_passages)
        response = chain.invoke({"question": query, "passages": passages})

        result = response["rerank"]
        reranked_passages = result.strip().split(reranking_template.separator)
        stripped_passages = [
            stripped_item
            for item in reranked_passages
            if (stripped_item := item.strip())
        ]

        return stripped_passages


class HFCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""

    # class Config:
    #     """Configuration for this pydantic object."""

    #     extra = Extra.forbid
    #     arbitrary_types_allowed = True
    
    # def compress_documents(
    #     self,
    #     documents: Sequence[Document],
    #     query: str,
    #     callbacks: Optional[Callbacks] = None,
    # ) -> Sequence[Document]:
    #     """
    #     Rerank documents using CrossEncoder.

    #     Args:
    #         documents: A sequence of documents to compress.
    #         query: The query to use for compressing the documents.
    #         callbacks: Callbacks to run during the compression process.

    #     Returns:
    #         A sequence of compressed documents.
    #     """
    #     scores = self.model.score([(query, doc.page_content) for doc in documents])
    #     docs_with_scores = list(zip(documents, scores))
    #     result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
    #     return [doc for doc, _ in result[: self.top_n]]

    @staticmethod
    def generate_response(self, 
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        
        scores = self.model.score([(query, doc.strip()) for doc in passages])
        docs_with_scores = list(zip(passages, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        # return [doc.strip() for doc, _ in result[: self.top_n]]
        return [doc for doc, _ in result[: keep_top_k]]

# class CrossEncoderReRanker:
#     def __init__(self):
#         self.model = SentenceTransformer(settings.CROSS_ENCODER_MODEL_ID)

#     def rerank(self, query: str, passages: list[str], keep_top_k: int) -> list[str]:
#         scores = self.model.predict([(query, passage) for passage in passages], show_progress_bar=False)
#         sorted_passages = [passage for _, passage in sorted(zip(scores, passages), reverse=True)]
        return sorted_passages[:keep_top_k]
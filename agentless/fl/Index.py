import copy
import os
from abc import ABC
import gc  # Add this to your imports

import tiktoken
from transformers import AutoTokenizer
from llama_index.core import (
    Document,
    MockEmbedding,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode

# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.index_skeleton import parse_global_stmt_from_code
from agentless.util.preprocess_data import (
    clean_method_left_space,
    get_full_file_paths_and_classes_and_functions,
)
from get_repo_structure.get_repo_structure import parse_python_file


def construct_file_meta_data(file_name: str, clazzes: list, functions: list) -> dict:
    meta_data = {
        "file_name": file_name,
    }
    meta_data["File Name"] = file_name

    if clazzes:
        meta_data["Classes"] = ", ".join([c["name"] for c in clazzes])
    if functions:
        meta_data["Functions"] = ", ".join([f["name"] for f in functions])

    return meta_data


def check_meta_data(meta_data: dict) -> bool:

    doc = Document(
        text="",
        metadata=meta_data,
        metadata_template="### {key}: {value}",
        text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
    )

    if (
        num_tokens_from_messages(
            doc.get_content(metadata_mode=MetadataMode.EMBED),
            model="text-embedding-3-small",
        )
        > Settings.chunk_size // 2
    ):
        # half of the chunk size should not be metadata
        return False

    return True


def build_file_documents_simple(
    clazzes: list, functions: list, file_name: str, file_content: str
) -> list[Document]:
    """
    Really simple file document format, where we put all content of a single file into a single document
    """
    documents = []

    meta_data = construct_file_meta_data(file_name, clazzes, functions)

    doc = Document(
        text=file_content,
        metadata=meta_data,
        metadata_template="### {key}: {value}",
        text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
    )
    doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
    doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
    if not check_meta_data(meta_data):
        # meta_data a bit too long, instead we just exclude meta data
        doc.excluded_embed_metadata_keys = list(meta_data.keys())
        doc.excluded_llm_metadata_keys = list(meta_data.keys())
        documents.append(doc)
    else:
        documents.append(doc)

    return documents


def build_file_documents_complex(
    clazzes: list, functions: list, file_name: str, file_content: str
) -> list[Document]:

    documents = []

    global_stmt, _ = parse_global_stmt_from_code(file_content)
    base_meta_data = construct_file_meta_data(file_name, clazzes, functions)

    for clazz in clazzes:
        content = "\n".join(clazz["text"])
        meta_data = copy.deepcopy(base_meta_data)
        meta_data["Class Name"] = clazz["name"]
        doc = Document(
            text=content,
            metadata=meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )

        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(meta_data):
            doc.excluded_embed_metadata_keys = list(meta_data.keys())
            doc.excluded_llm_metadata_keys = list(meta_data.keys())
        documents.append(doc)

        for class_method in clazz["methods"]:
            method_meta_data = copy.deepcopy(base_meta_data)
            method_meta_data["Class Name"] = clazz["name"]
            method_meta_data["Method Name"] = class_method["name"]
            content = clean_method_left_space("\n".join(class_method["text"]))

            doc = Document(
                text=content,
                metadata=method_meta_data,
                metadata_template="### {key}: {value}",
                text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
            )
            doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
            doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
            if not check_meta_data(method_meta_data):
                doc.excluded_embed_metadata_keys = list(method_meta_data.keys())
                doc.excluded_llm_metadata_keys = list(method_meta_data.keys())
            documents.append(doc)

    for function in functions:
        content = "\n".join(function["text"])
        function_meta_data = copy.deepcopy(base_meta_data)
        function_meta_data["Function Name"] = function["name"]
        doc = Document(
            text=content,
            metadata=function_meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )

        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(function_meta_data):
            doc.excluded_embed_metadata_keys = list(function_meta_data.keys())
            doc.excluded_llm_metadata_keys = list(function_meta_data.keys())
        documents.append(doc)

    if global_stmt != "":
        content = global_stmt
        global_meta_data = copy.deepcopy(base_meta_data)

        doc = Document(
            text=content,
            metadata=global_meta_data,
            metadata_template="### {key}: {value}",
            text_template="Metadata:\n{metadata_str}\n-----\nCode:\n{content}",
        )
        doc.excluded_embed_metadata_keys = ["file_name"]  # used for searching only.
        doc.excluded_llm_metadata_keys = ["file_name"]  # used for searching only.
        if not check_meta_data(global_meta_data):
            doc.excluded_embed_metadata_keys = list(global_meta_data.keys())
            doc.excluded_llm_metadata_keys = list(global_meta_data.keys())
        documents.append(doc)

    return documents


class EmbeddingIndex(ABC):
    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        persist_dir,
        filter_type,
        index_type,
        chunk_size,
        chunk_overlap,
        logger,
        **kwargs,
    ):
        self.instance_id = instance_id
        self.structure = structure
        self.problem_statement = problem_statement
        self.persist_dir = persist_dir + "/{instance_id}"
        self.filter_type = filter_type
        self.index_type = index_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        self.kwargs = kwargs
        # set some embedding global settings.

        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

    def filter_files(self, files):
        if self.filter_type == "given_files":
            given_files = self.kwargs["given_files"][: self.kwargs["filter_top_n"]]
            return given_files
        elif self.filter_type == "none":
            # all files are included
            return [file_content[0] for file_content in files]
        else:
            raise NotImplementedError

    def retrieve(self, mock=False):

        persist_dir = self.persist_dir.format(instance_id=self.instance_id)
        # token_counter = TokenCountingHandler(
        #    tokenizer=tiktoken.encoding_for_model("BAAI/bge-small-en-v1.5").encode
        # )
        hf_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

        # 2. Use the Hugging Face tokenizer's encode method
        token_counter = TokenCountingHandler(tokenizer=hf_tokenizer.encode)
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", embed_batch_size=10, device="cpu"
        )
        Settings.embed_model = embed_model
        if not os.path.exists(persist_dir) or mock:
            files, _, _ = get_full_file_paths_and_classes_and_functions(self.structure)
            filtered_files = self.filter_files(files)

            self.logger.info(f"Total number of considered files: {len(filtered_files)}")

            # --- CHANGED: Incremental Indexing Strategy ---
            index = None
            current_batch = []
            BATCH_SIZE = 20  # Keep this small. Only hold 20 docs in RAM at once.

            total_docs_processed = 0

            for file_content in files:
                file_name = file_content[0]
                if file_name not in filtered_files:
                    continue

                content = "\n".join(file_content[1])

                # Create documents for this ONE file
                class_info, function_names, _ = parse_python_file(None, content)

                if self.index_type == "simple":
                    docs = build_file_documents_simple(
                        class_info, function_names, file_name, content
                    )
                elif self.index_type == "complex":
                    docs = build_file_documents_complex(
                        class_info, function_names, file_name, content
                    )
                else:
                    raise NotImplementedError

                current_batch.extend(docs)

                # If our batch is full, push to index and CLEAR memory
                if len(current_batch) >= BATCH_SIZE:
                    if index is None:
                        # First batch creates the index
                        index = VectorStoreIndex.from_documents(
                            current_batch, embed_model=embed_model
                        )
                    else:
                        # Subsequent batches are inserted
                        index.insert_nodes(current_batch)

                    total_docs_processed += len(current_batch)
                    print(f"Processed {total_docs_processed} documents...")

                    # WIPE MEMORY
                    current_batch = []
                    gc.collect()  # Force Python to release RAM now

            # Process any remaining documents in the final batch
            if current_batch:
                if index is None:
                    index = VectorStoreIndex.from_documents(
                        current_batch, embed_model=embed_model
                    )
                else:
                    index.insert_nodes(current_batch)
                total_docs_processed += len(current_batch)

            self.logger.info(f"Total number of documents: {total_docs_processed}")

            if mock:
                # Mocking logic (unchanged, just ensure index is valid)
                embed_model = MockEmbedding(embed_dim=1024)
                Settings.callback_manager = CallbackManager([token_counter])
                # If we mocked, we likely didn't run the loop above,
                # so you might need to handle the 'index is None' case for pure mock runs.
                if index is None:
                    index = VectorStoreIndex.from_documents([], embed_model=embed_model)

            # Persist the final built index
            index.storage_context.persist(persist_dir=persist_dir)

        else:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)

        self.logger.info(f"Retrieving with query:\n{self.problem_statement}")

        retriever = VectorIndexRetriever(
            index=index, similarity_top_k=100, embed_model=embed_model
        )
        documents = retriever.retrieve(self.problem_statement)
        # print(f"RETRIEVED: {documents}")

        self.logger.info(
            f"Embedding Tokens: {token_counter.total_embedding_token_count}"
        )
        print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")

        traj = {
            "usage": {"embedding_tokens": token_counter.total_embedding_token_count}
        }

        token_counter.reset_counts()

        if mock:
            self.logger.info("Skipping since mock=True")
            return [], None, traj

        file_names = []
        meta_infos = []

        for node in documents:
            file_name = node.node.metadata["File Name"]
            if file_name not in file_names:
                file_names.append(file_name)
                self.logger.info("================")
                self.logger.info(file_name)

            self.logger.info(node.node.text)

            meta_infos.append({"code": node.node.text, "metadata": node.node.metadata})

        return file_names, meta_infos, traj

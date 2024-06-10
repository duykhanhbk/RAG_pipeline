# RAG_pipeline
RAG with advanced technique

### The data preprocessing pipeline

- **Azure Document Intelligence**: https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence
An AI service that applies advanced machine learning to extract text, key-value pairs, tables, and structures from documents automatically and accurately
- **Customize Multi-modal parser (./preprocessing/multimodal_parser)**: 
    - Audio files: transcribed use Whisper (use in-house or OpenAI API)
    - image files: use gpt-4o
    - Video Files: Both the transcripts and various frame slices are processed in the same manner, with each being chunked and embedded


### Vector DB: Qdrant + Hybrid search

**Why Qdrant ?:**
- Qdrant enhances search, offering semantic, similarity, multimodal, and hybrid search capabilities for accurate, user-centric results, serving applications in different industries like e-commerce to healthcare.
- It is built in Rust.
- Apache-2.0 license — open-source 🔥
- It has a great and intuitive Python SDK.
- It has a freemium self-hosted version to build PoCs for free.
- It supports unlimited document sizes, and vector dims of up to 645536.
- It is production-ready. Companies such as Disney, Mozilla, and Microsoft already use it.
- It is one of the most popular vector DBs out there.

**Why do we still need keyword search?**
A keyword-based search was the obvious choice for search engines in the past. It struggled with some common issues, but since we didn’t have any alternatives, we had to overcome them with additional preprocessing of the documents and queries. Vector search turned out to be a breakthrough, as it has some clear advantages in the following scenarios:

- 🌍 Multi-lingual & multi-modal search
- 🤔 For short texts with typos and ambiguous content-dependent meanings
- 👨‍🔬 Specialized domains with tuned encoder models
- 📄 Document-as-a-Query similarity search

It doesn’t mean we do not keyword search anymore. There are also some cases in which this kind of method might be useful:

- 🌐💭 Out-of-domain search. Words are just words, no matter what they mean. BM25 ranking represents the universal property of the natural language - less frequent words are more important, as they carry most of the meaning.
- ⌨️💨 Search-as-you-type, when there are only a few characters types in, and we cannot use vector search yet.
- 🎯🔍 Exact phrase matching when we want to find the occurrences of a specific term in the documents. That’s especially useful for names of the products, people, part numbers, etc.

There is not a single definition of hybrid search. Actually, if we use more than one search algorithm, it might be described as some sort of hybrid. Some of the most popular definitions are:

- A combination of vector search with attribute filtering. We won’t dive much into details, as we like to call it just filtered vector search.
- Vector search with keyword-based search. This one is covered in this article.
- A mix of dense and sparse vectors. That strategy will be covered in the upcoming article.

**How to deploy Qdrant in local:**

First, download the latest Qdrant image from Dockerhub:

```docker pull qdrant/qdrant```

Then, run the service:

```docker run -p 6333:6333 -p 6334:6334
    -v $(pwd)/qdrant_storage:/qdrant/storage:z 
    qdrant/qdrant```

Under the default configuration all data will be stored in the ./qdrant_storage directory. This will also be the only directory that both the Container and the host machine can both see.

Qdrant is now accessible:

- REST API: localhost:6333
- Web UI: localhost:6333/dashboard
- GRPC API: localhost:6334

ref: https://qdrant.tech/documentation/quick-start/

### Guide to Choosing the Best Embedding Model for Your Application

ref: https://weaviate.io/blog/how-to-choose-an-embedding-model

- **Step 1**: Identify your use case: Modality (e.g., text only or multimodal), subject domain (e.g., coding, law, medical, multilingual, etc.), and deployment mode? 
- **Step 2**: Choose a baseline model: Based on MTEB leaderboard is a good starting point (Task, Score, model size and memory usage, embedding dimensions, max_tokens,..)
- **Step 3**: Evaluate model on your use-case: Prepare evaluation dataset with pair (question, gold_context) Metric: recision, recall, MRR, MAP, and NDCG
- **Step 4**: Iterate step 1 → step 3 to find best embedding candidate and finetuning if needed


### Advanced RAG Technique

**Good RAG requires good retrieval, and good retrieval requires a good query understanding layer.**

- Query Expansion
- Self Query
- Hybrid & Filtered vector search
- Query Rewriting

**1. Query Expansion:**

**The problem**
- In a typical retrieval step, you query your vector DB using a single point.
- The issue with that approach is that by using a single vector, you cover only a small area of your embedding space.
- Thus, if your embedding doesn't contain all the required information, your retrieved context will not be relevant.

**The solution:**
- You use an LLM to generate multiple queries based on your initial query.
- These queries should contain multiple perspectives of the initial query.
- Thus, when embedded, they hit different areas of your embedding space that are still relevant to our initial question.

**2. Self Query:**

**The problem:**
- When embedding your query, you cannot guarantee that all the aspects required by your use case are present in the embedding vector.
- For example, you want to be 100% sure that your retrieval relies on the tags provided in the query.
- The issue is that by embedding the query prompt, you can never be sure that the tags are represented in the embedding vector or have enough signal when computing the distance against other vectors.

**The solution:**
- What if you could extract the tags within the query and use them along the embedded query? That is what self-query is all about!
- You use an LLM to extract various metadata fields that are critical for your business use case (e.g., tags, author ID, number of comments, likes, shares, etc.)

**3. Hybrid & Filtered vector search:**

**The problem**
- Embeddings are great for capturing the general semantics of a specific chunk. But they are not that great for querying specific keywords.
- For example, if we want to retrieve article chunks about LLMs from our Qdrant vector DB, embeddings would be enough. However, if we want to query for a specific LLM type (e.g., LLama 3), using only similarities between embeddings won’t be enough.
- Thus, embeddings are not great for finding exact phrase matching for specific terms.

**The solution:**
Combine the vector search technique with one (or more) complementary search strategy, which works great for finding exact words.

**Note:** Cross-encoder takes a pair of texts and predicts the similarity of them. Unlike embedding models, cross-encoders do not compress text into vector, but uses interactions between individual tokens of both texts. In general, they are more powerful than both BM25 and vector search, but they are also way slower. That makes it feasible to use cross-encoders only for re-ranking of some preselected candidates.

**Reranking**
- Cross encoder model: Huggingface
- LLM Reranker: GPT4, GPT-4o, etc
- Cohere reranker API

**4. Query rewriting:**

- Sub-question decomposition: Break a complex question into sub-questions. Unlike pure chain of thought, you can break a question down into a parallelizable sub-questions that you can try answering all at once.
- HyDE: rewrite the question to hallucinate an answer that better aligns with the embedding semantics.
- Step-back prompting: To answer a complex question, take a “step back” and answer a more generic question to better answer the specific one.
- ref: https://generativeai.pub/advanced-rag-retrieval-strategy-query-rewriting-a1dd61815ff0


### LLM Finetuning && Embedding Finetuning

**Embedding finetuning && Data synthetic geneeration**: https://www.philschmid.de/fine-tune-embedding-model-for-rag

Step by step (Modification by Kevin):
- Create & Prepare embedding dataset: Data synthetic
- Bonus: Create hard negative samples
- Create baseline and evaluate pretrained model
- Define loss function: Matryoshka Representation, Constrastive Loss
- Fine-tune embedding model with SentenceTransformersTrainer or our custumize code
- Evaluate fine-tuned model against baseline

**LLM finetuning with QA context task**

Step by step:
- Create & Prepare QA task with context (chunking): Data synthetic using GPT-4o, Gemini, etc
- Bonus: CoT (Chain of Thought) QA data generation
- Finetuning: LoRA

### LLM serving && Embedding Serving

- LLMDeploy: more thoughput compare with vllm (x1.8)
- vllm
- TensorRT-LLM

**How to SERVING LLMDeploy WITH OPENAI COMPATIBLE SERVER**

Option 1: Launching with lmdeploy CLI

```lmdeploy serve api_server meta-llama/Meta-Llama-3-8B-Instruct --server-port 23333```

Option 2: Deploying with docker

With LMDeploy official docker image, you can run OpenAI compatible server as follows:

```docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server meta-llama/Meta-Llama-3-8B-Instruct```

ref: https://lmdeploy.readthedocs.io/en/latest/serving/api_server.html

**How to Serving LLM in-house with vllm**

ref: https://docs.vllm.ai/en/stable/

vLLM offers official docker image for deployment. The image can be used to run OpenAI compatible server. The image is available on Docker Hub as vllm/vllm-openai.

```docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1```

**How to serving Embedding model and CrossEncoder Reranker?**

Option 1: Infinity

Infinity is a high-throughput, low-latency REST API for serving vector embeddings, supporting all sentence-transformer models and frameworks. Infinity is developed under MIT License. Infinity powers inference behind Gradient.ai and other Embedding API providers.

Why Infinity?

Infinity provides the following features:

- Deploy any model from MTEB: deploy the model you know from SentenceTransformers
- Fast inference backends: The inference server is built on top of torch, optimum(onnx/tensorrt) and CTranslate2, using FlashAttention to get the most out of CUDA, ROCM, CPU or MPS device.
- Dynamic batching: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your device as soon as ready. Similar max throughput on GPU as text-embeddings-inference.
- Correct and tested implementation: Unit and end-to-end tested. Embeddings via infinity are identical to SentenceTransformers (up to numerical precision). Lets API users create embeddings till infinity and beyond.
- Easy to use: The API is built on top of FastAPI, Swagger makes it fully documented. API are aligned to OpenAI's Embedding specs. See below on how to get started.

**Deploy with Docker**

```port=7997
model1=michaelfeil/bge-small-en-v1.5
model2=mixedbread-ai/mxbai-rerank-xsmall-v1
volume=$PWD/data

docker run -it --gpus all \
 -v $volume:/app/.cache \
 -p $port:$port \
 michaelf34/infinity:latest \
 v2 \
 --model-id $model1 \
 --model-id $model2 \
 --port $port```

The cache path at inside the docker container is set by the environment variable HF_HOME

ref:
- https://github.com/michaelfeil/infinity
- https://michaelfeil.eu/infinity/0.0.41/

Option 2: vllm embedding server

### Evaluation pipeline

- LLM (GPT4, GPT-4o, Gemini) + Prompt Evaluation
- RAGAS:
    - Retriever based: Context relevency, Context recall, 
    - End-to-end: Answer similarity (compare with gold answer); Answer correctness
- Compare LLM and Embedding model: 
    - https://docs.ragas.io/en/stable/howtos/applications/compare_embeddings.html

    - https://docs.ragas.io/en/stable/howtos/applications/compare_llms.html 


### Monitoring/Experiment Tracker

- Use CometLLM Tracker: prompt monitoring dashboard; experiment tracker, model registry
- Use a bigger LLM (e.g., GPT4, GPT-4o) to evaluate the results of our LLM (API or in-house) and fine-tuned LLM (opensource, example: LLama3, Mistral, Qwen, Mixtral, Deepseek)


### How to run: TO DO (UPDATE !!)

- You need setup Qdrant service and stored all data you need
- Deploy with Docker and docker-compose
- Simple UI for testing: streamlit

1. **Build and run the services**:
   In the directory containing your `Dockerfile` and `docker-compose.yml`, run:

   ```sh
   docker-compose up --build
   ```

2. **Access the Streamlit app**:
   Open your web browser and navigate to `http://localhost:8501` to access the Streamlit interface.

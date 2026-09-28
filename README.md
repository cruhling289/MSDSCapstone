# **A RAG System For Student Q&A on MSDS Capstone**


## **Overview**

This project is a Retrieval-Augmented Generation (RAG) system designed to answer questions about the UVA School of Data Science Capstone program.

The system retrieves relevant information from two sources:



1. **Capstone Day Planning** — information about the Capstone Day event, including the event schedule, attendance, awards, and venue.
2. **Sponsor Experience** — information from the UVA School of Data Science website about sponsoring a capstone project, including sponsor benefits, responsibilities, and expected deliverables.

The retrieved information is passed to a hosted language model (via the Groq API) to generate an answer to the user's question.


## **How It Works**

Source Documents

      ↓

Text Extraction (nav/header/footer stripped from the webpage)

      ↓

Chunking

      ↓

Embeddings (all-MiniLM-L6-v2)

      ↓

Semantic Search (cosine similarity)

      ↓

Relevant Chunks

      ↓

Groq API (openai/gpt-oss-20b)

      ↓

Generated Answer


### **1. Data Collection**

The project collects information from:



* A Markdown file stored in the GitHub repository: `data/capstone_day_planning.md`
* The UVA School of Data Science Sponsor Experience webpage: https://datascience.virginia.edu/data-science-capstone/sponsor-experience

The Markdown file is downloaded directly from GitHub using `requests`. The webpage is also fetched using `requests`, then parsed with Beautiful Soup. Navigation, header, footer, script, style, and form elements are stripped out before extracting text — the raw page includes a large amount of site-wide navigation and a newsletter signup form that would otherwise pollute the chunk set with irrelevant content.


### **2. Combining the Sources**

The cleaned text from both sources is combined into one variable so retrieval can search across both at once:

text = text1 + "\n\n" + web_text


### **3. Chunking**

The combined text is split into overlapping chunks:



* Chunk size: **1,200 characters**
* Overlap: **20 characters**


### **4. Embeddings**

Each chunk is converted into a numerical vector using `all-MiniLM-L6-v2` (384-dimensional embeddings), via Sentence Transformers. All chunks are embedded in a single batched call rather than one at a time.


### **5. Retrieval**

The user's question is embedded with the same model, and `sentence_transformers.util.semantic_search` is used to find the **K = 4** most relevant chunks by cosine similarity. Retrieving more than one or two chunks matters here — some answers (e.g. sponsorship benefits) span multiple non-adjacent chunks, and a smaller K risks leaving out real content.


### **6. Generation**

The retrieved chunks are inserted into a prompt instructing the model to answer using only that context, and the prompt is sent to the **Groq API** (currently `openai/gpt-oss-20b`) for generation. Groq was chosen over running a model locally because it eliminates model-download and model-loading time from every run — the earlier local-Qwen version spent most of its runtime just loading model weights.

Note: `openai/gpt-oss-20b` is a reasoning model — it can spend part of its token budget on internal "thinking" before producing the visible answer. `reasoning_effort="low"` is used to keep that overhead small, since this task is straightforward context lookup rather than multi-step reasoning.


## **Technologies**



* **Python**
* **Google Colab**
* **GitHub**
* **Requests**
* **Beautiful Soup**
* **Sentence Transformers** (`all-MiniLM-L6-v2`)
* **scikit-learn / sentence-transformers <code>util.semantic_search</code></strong>
* <strong>Groq API</strong> (<code>openai/gpt-oss-20b</code>)


## **Project Structure**

MSDSCapstone/

│

├── data/

│   └── capstone_day_planning.md

│

├── notebooks/

│   └── MSDSCapstone.ipynb

│

└── README.md


## **Notebook Structure**

The notebook is split into two cells to separate one-time setup cost from per-question cost:



* **Setup cell** (run once per Colab session): downloads/cleans the source data, builds and embeds all chunks, and creates the Groq client. This is the only slow part of the pipeline (a few seconds for data collection and embedding).
* **Query cell** (run once per question): takes a question, retrieves the top-K chunks, and calls the Groq API for an answer. This is the fast, repeatable part — no model loading involved, since generation happens on Groq's hosted infrastructure.


## **Configuration**


<table>
  <tr>
   <td><strong>Variable</strong>
   </td>
   <td><strong>Value</strong>
   </td>
   <td><strong>Purpose</strong>
   </td>
  </tr>
  <tr>
   <td><code>K</code>
   </td>
   <td>6
   </td>
   <td>Number of chunks retrieved per question
   </td>
  </tr>
  <tr>
   <td><code>CHUNK_SIZE</code>
   </td>
   <td>1200
   </td>
   <td>Characters per chunk
   </td>
  </tr>
  <tr>
   <td><code>OVERLAP</code>
   </td>
   <td>20
   </td>
   <td>Character overlap between chunks
   </td>
  </tr>
  <tr>
   <td><code>MAX_NEW_TOKENS</code>
   </td>
   <td>700
   </td>
   <td>Response length ceiling
   </td>
  </tr>
  <tr>
   <td><code>reasoning_effort</code>
   </td>
   <td><code>"low"</code>
   </td>
   <td>Minimizes internal reasoning tokens on <code>openai/gpt-oss-20b</code>
   </td>
  </tr>
</table>



## **Example Questions**


### **Capstone Day**



* Which talks happen before 11 AM?
* Where is Capstone Day held?
* What awards are given?
* What is the presentation schedule?


### **Sponsor Experience**



* What are the benefits of sponsoring a capstone?
* What responsibilities does a sponsor have?
* How often do sponsors meet with student teams?
* What deliverables do capstone teams provide?


## **Known Limitations**



* **Model hallucination**: smaller/faster models can occasionally continue past the real answer content (e.g., inventing extra list items) if given more tokens than the answer needs. The system prompt explicitly instructs the model to answer only from the provided context, which helps but isn't foolproof — this is an active area of refinement.
* **Groq model availability**: Groq periodically deprecates and replaces model IDs (this project has already migrated once, from `mixtral-8x7b-32768`, which was decommissioned). If a `model_decommissioned` error appears, check https://console.groq.com/docs/deprecations for the current recommended replacement.


## **Prior Approach**

An earlier version of this project ran generation locally using **Qwen3-1.7B** via `transformers`, loading model weights directly in Colab. This was replaced with the Groq API because local model loading (downloading and initializing ~3.4GB of weights) dominated total runtime, often taking over a minute per run regardless of GPU availability.


## **Purpose**

This project was created as part of a data science capstone project to understand how Retrieval-Augmented Generation systems work and how information from multiple sources can be retrieved and used by a language model to answer questions.

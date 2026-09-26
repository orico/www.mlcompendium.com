# Natural Language Processing (NLP)

Language starts as strings, and the transformer mechanism is already behind the reader. The chapter therefore walks up from raw text: first the foundations that turn strings into something countable, then the tasks people actually build, and last the neural models that replace hand-built features with pretrained ones.
After this chapter the reader can go from matching and TF-IDF to a pretrained text model and a decoding step.

## Foundations

Before any model, it helps to know how little metric progress many NLP algorithms show once datasets are normalized, and [A Reality Check](a-reality-check.md) is that warning. With expectations set, [Foundation NLP](foundation-nlp.md) gathers the basic pipelines: chunking, collocations, stemming, phrase models, and Hebrew tools. The libraries that run those pipelines, spaCy compared with the others, plus datasets and embedding repositories, are in [NLP Tools](nlp.md).

The first concrete problem on strings is matching them. [Name Matching](name-matching.md) handles fuzzy person-name matching with articles, public name datasets, and libraries, and [String Matching](string-matching.md) widens that to fuzzy string matching in general and regex tooling. Once text has to be weighted rather than matched, [TF-IDF](tf-idf.md) defines term frequency and inverse document frequency and the sparse-text trick of weighting word vectors with IDF. Those weights feed [Topics Modeling](topics-modeling.md), the reading list that walks LSA, NMF, LDA, Mallet, coherence, lda2vec, and Top2Vec. The last foundation is how text becomes model input at all: [Tokenization](tokenization.md) covers BPE and tiktoken, Hugging Face tokenizers, and a BERT tokenizer guide.

## Tasks

With text represented, the chapter turns to the jobs built on it. [Named Entity Recognition (NER)](named-entity-recognition-ner.md) collects milestone NER papers plus spaCy, SNER, and BiLSTM-CRF style tutorials. [Sentiment Analysis](sentiment-analysis.md) lists sentiment databases, ground-truth rating and annotator-agreement practice, and analyzers such as VADER and TextBlob. [Question Answering](question-answering.md) covers building Q&A systems, image Q&A, and retrieval backed by Milvus, and [Summarization](summarization.md) covers extractive and abstractive methods, including TextRank-style approaches.

Some tasks are about the language itself: [Language Detection Identification Generation (NLD, NLI, NLG)](language-detection-identification-generation-nld-nli-nlg.md) covers language ID tools, neural language models, generation, translation, BLEU, and transliteration. Retrieval gets its own page in [SEARCH](search.md), which calls out BERT-based search patterns and the recurring semantic-search problems with their fixes. [Intent Recognition](intent-recognition.md) closes the tasks with what intent classification is and how it shows up in chatbot-style pipelines.

## Neural models

The tasks above can be solved with counts and rules, but the neural path is where the chapter ends. [Neural NLP](neural-nlp.md) groups CNN-for-text material and sequence-to-sequence resources. [Language Embeddings](language-embeddings.md) is the reading list from foundation word vectors through sentence and document models, and [Pretrained Language Models](pretrained-language-models.md) moves to ELMo, ULMFiT, BERT, XLNet, GPT-2, and GPT-3 with their transfer-learning and fine-tuning guides. A generative model still needs a way to pick its output, and [Decoding Algorithms For NLP](decoding-algorithms-for-nlp.md) walks through greedy decoding, beam search, and sampling.

The remaining pages stretch those models. [Multi Language](multi-language.md) points at multilingual sentence embeddings such as LASER and cross-lingual language-model pretraining. [Augmentation](augmentation.md) lists ways to grow text data: synonym swaps, embedding neighbors, back translation, and generation. [Knowledge Graphs](knowledge-graphs.md) is about building knowledge graphs from text and structured sources, including automatic creation from text with spaCy.

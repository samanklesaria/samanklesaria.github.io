## Topic Extraction: Old and New

How do you find thematic clusters in a large corpus of text documents? There are the standared algorithms baked into `sklearn`: k-means, nonnegative matrix factorization and LDA. But contemporary NLP has largely moved on from bag-of-words representations. Can I get better results with some pretrained transformer models? In this notebook, I'll be playing around with topic extraction ideas using some models from huggingface. 


```python
import numpy as np
from typing import Optional
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import AutoTokenizer, BartForConditionalGeneration, AutoModel
from sklearn.metrics import silhouette_samples, silhouette_score
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
```

## Fetching the Data

For demonstration purposes, I'll use a few categories from the standard 20-newsgroups dataset.


```python
categories = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    categories=categories,
    shuffle=True,
    random_state=42,
)
```

Some of the documents in the dataset are only a few words; I only want to deal with documents that are least a couple hundred characters. 


```python
filtered_text = filter(lambda x: len(x) > 200, (d.strip() for d in dataset.data))
```


```python
X = list(filtered_text)
```

It will be convenient to use numpy-style indexing later on.


```python
np_text = np.array(X)
```

## LDA

As a baseline, we can use the standard Latent Dirichlet Allocation model. Each topic has its own distribution over words. Each document has its own distribution over topics. We observe documents as bags of words and do maximum likelihood estimation.


```python
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words="english")
```


```python
tf = tf_vectorizer.fit_transform(X)
```


```python
lda = LatentDirichletAllocation(
    n_components=8,
    max_iter=10,
    learning_method="online",
    learning_offset=50.0
)
```


```python
lda.fit(tf)
```


```python
tf_feature_names = tf_vectorizer.get_feature_names_out()
```


```python
def top_components(terms, m, n=5, k=5):
    ixs = np.argsort(-np.abs(m.components_[:n]), axis=-1)
    return [[(float(a), b) for (a,b) in
          zip(np.round(m.components_[i, ix[:k]], decimals=2), terms[ixs[i, :k]])]
          for (i, ix) in enumerate(ixs)]
```

This recovers categories for *space* and *graphics* like we expect. But we're not seeing much in the way of atheism or religion. 


```python
top_components(tf_feature_names, lda)
```




    [[(1283.08, 'space'),
      (456.0, 'nasa'),
      (420.76, 'earth'),
      (373.97, 'launch'),
      (305.22, 'orbit')],
     [(138.5, 'theory'),
      (119.63, 'gamma'),
      (114.11, 'universe'),
      (50.56, 'black'),
      (45.2, 'matter')],
     [(1233.8, 'god'),
      (1114.04, 'people'),
      (922.51, 'don'),
      (854.54, 'think'),
      (836.6, 'just')],
     [(1160.77, 'image'),
      (906.21, 'edu'),
      (824.99, 'graphics'),
      (793.88, 'jpeg'),
      (622.48, 'file')],
     [(37.06, 'van'),
      (28.58, 'command'),
      (20.55, 'op'),
      (18.57, 'prof'),
      (15.6, 'timer')]]



## NMF

Another off-the shelf approach is to use nonnegative matrix factorization of tfidf features. 


```python
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, stop_words="english")
```


```python
X_tfidf = tfidf_vectorizer.fit_transform(X)
```


```python
tfidf_terms = tfidf_vectorizer.get_feature_names_out()
```


```python
nmf = NMF(n_components=4, alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1).fit(X_tfidf)
```

Here, we see strong evidence for topics about Christianity, space and images. It's a little vague, though. For example, we don't differentiate atheism and theism, and we don't see anything about computers in the graphics topic. 


```python
top_components(tfidf_terms, nmf)
```




    [[(9.43, 'god'),
      (0.53, 'jesus'),
      (0.5, 'bible'),
      (0.43, 'believe'),
      (0.0, '00')],
     [(6.41, 'image'),
      (0.5, 'images'),
      (0.0, '00'),
      (0.0, 'pbmplus'),
      (0.0, 'pbm')],
     [(8.98, 'space'),
      (0.56, 'nasa'),
      (0.0, '00'),
      (0.0, 'pbmplus'),
      (0.0, 'pbm')],
     [(0.9, 'ico'),
      (0.88, 'bobbe'),
      (0.88, 'bronx'),
      (0.87, 'queens'),
      (0.87, 'beauchaine')]]



## KMeans via SVD

Rounding out our classical approaches is the venerable k-means algorithm. We'll use tf-idf features with dimensionality reduced via SVD. 


```python
lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
```


```python
X_lsa = lsa.fit_transform(X_tfidf)
```


```python
lsa[0].explained_variance_ratio_.sum()
```




    np.float64(0.19657892886841927)




```python
kmeans = KMeans(n_clusters=4, n_init=20)
kmeans.fit(X_lsa)
```


```python
original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
```

This gives pretty good results: we get Christianity, computer graphics, and space! We just can't tell atheism apart from Christianity. 


```python
terms[order_centroids[:, :5]]
```




    array([['don', 'people', 'think', 'just', 'say'],
           ['space', 'orbit', 'launch', 'nasa', 'shuttle'],
           ['graphics', 'image', 'thanks', 'program', 'files'],
           ['god', 'jesus', 'bible', 'believe', 'christian']], dtype=object)



## Summarization with Transformers

Instead of describing topics as multinomial distributions of words, let's see if we can get a transformer model to generate human readable summaries instead!


```python
bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to("mps")
```


```python
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
```

The examples in the Huggingface documentation encourage you to use the `bart-large-cnn` model without a decoder prompt, but I find the summaries are better with a little prompting. Browsing the huggingbase codebase shows an undocumented `decoder_input_ids` option for precisely this purpose.


```python
gtoks = bart_tokenizer("The major themes in this documents are ", truncation=True, return_tensors="pt", max_length=1024)['input_ids'].to('mps')
```


```python
def summarize(strs):
    toks = bart_tokenizer(strs, padding="longest", truncation=True, return_tensors="pt", max_length=1024)
    expanded = gtoks.expand(toks['input_ids'].shape[0], -1)
    return bart_tokenizer.batch_decode(
        bart.generate(**toks.to("mps"), decoder_input_ids=expanded), skip_special_tokens=True)
```

## Summarized Central Texts

We can start by just summarizing the documents nearest to the center of each cluster. 


```python
def most_central(X, kmeans):
    return [np.argmin(((X[kmeans.labels_ == i] - c)**2).sum(axis=-1))
        for i, c in enumerate(kmeans.cluster_centers_)]
```


```python
centers = most_central(X_lsa, kmeans)
central_texts = [str(np_text[kmeans.labels_ == i][a]) for i, a in enumerate(centers)]
```


```python
summarize(central_texts)
```




    ["The major themes in this document are ativism and moral relativism. The author also discusses the Rodney King case and the bombing of Dresden in World War 2. The final section of the article is a collection of the author's comments on the case, as well as a selection of his other writings.",
     "The major themes in this document are atation: microgravity, life sciences research, and spacecraft maintenence. SSF's 3 main functions require quite different environments and are prime candidates for constellization. We need to study life sciences not just in microgravity but also in lunar and Martian gravities, and in the radiation environments of deep space.",
     'The major themes in this document are i graphics, raster graphics, and vector graphics. There are also general references for graphics questions. The latest version of this FAQ is always available on the archive site pit-manager.mit.edu (alias rtfm.mit)',
     "The major themes in this document are a, a love for God, and a fear of God. The author of this article, Brian K., says he doesn't see an open mind in Brian's postings. He says Brian could be a St. Paul, who mocked Christians as you do, but was saved by God."]



This seems to get at the theism/ atheism distinction a bit!

## Multi Document Summarization using LogitsProcessor

Perhaps relying on a single representative document per class is too restrictive. We can let the probability of generating a token given a set of context vectors be the product of the token's probability in each context. The intuition is that a token is a good choice for describing a cluster if it's a good choice for describing each document within the cluster individually. This is easy to accomplish with a little abuse of huggingface's `logits_processor` argument. 


```python
def multiply_logits(input_ids, scores):
    return scores.log_softmax(dim=-1).sum(dim=0, keepdim=True)
```


```python
def summarize_group(strs, processor=multiply_logits):
    toks = bart_tokenizer(strs, padding="longest", truncation=True, return_tensors="pt", max_length=1024)
    expanded = gtoks.expand(toks['input_ids'].shape[0], -1)
    output = bart.generate(**toks.to("mps"), decoder_input_ids=expanded, num_beams=1, logits_processor=[processor])
    return bart_tokenizer.batch_decode(output[0][None], skip_special_tokens=True)
```


```python
def top_per_cluster(kmeans, X):
    return [np_text[kmeans.labels_ == i][np.argsort(((X[kmeans.labels_ == i] - c)**2).sum(axis=-1))[:16]]
        for i, c in enumerate(kmeans.cluster_centers_)]
```

Alas, this doesn't seem to work particularly well. 


```python
top_lsa = top_per_cluster(kmeans, X_lsa)
[summarize_group([str(a) for a in c])[0] for c in top_lsa]
```




    ['The major themes in these documents are i, the "world of the mind" and the "life of the soul" The author also discusses the "real" meaning of the word "soul" and "life" The "world" of the "sociopath" is the "human" world, the author says.',
     'The major themes in these documents are ations of the space program. The main focus of the articles is the space shuttle program. This article includes a list of the most popular articles on the topic. The article also includes a selection of the best articles on space.',
     'The major themes in these documents are i, psi, and png. The main purpose of this page is to help people understand the various themes in the software. The software is designed to be used on computers with a high level of performance. The program is designed for the PC, Mac, and PC-i.',
     'The major themes in these documents are i, the God-given right to be a god, and the God of the Bible. The author also discusses the role of the church in the culture of the world. The book is published by Oxford University Press. The website is free to read.']



Retrieval augmented generation does something similar, marginalizing each generated token's distribution over all possible context documents. Perhaps the same approach could work here?


```python
def marginalize_logits(input_ids, scores):
    return scores.softmax(dim=-1).sum(dim=0, keepdim=True).log()
```

Nope: results are similarly bad. 


```python
[summarize_group([str(a) for a in c], processor=marginalize_logits)[0] for c in top_lsa]
```




    ['The major themes in this documents are -ism, theory of the mind, and the concept of God. The author also discusses the role of religion in the culture of the U.S. and the role that religion plays in the U-S. relationship.',
     'The major themes in this documents are atmosphere, space, and the universe. The list of topics is broken down into three categories: space, space exploration, and space exploration. The first section of the report is a list of the topics that have been discussed in the past.',
     'The major themes in this documents are i, the computer graphics language, and the fractal language. The project is the work of Nikolaus H. F. Fotis. The program is written in C. The code is written by F. H. Haines.',
     'The major themes in this documents are i, the God-given right to believe and the God of the Bible. The author says that the Bible is not a "set of rules" but a "lifestyle" guide. The book is published by Oxford University Press.']



## Neural Embeddings

We can also use transformers for the document embeddings to be clustered with k-means. The following model is another BERT variant fine-tuned for generating embeddings. 


```python
minilm_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
minilm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to('mps')
```


```python
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```


```python
loader = DataLoader(X, batch_size=16)
```


```python
embeddings = []
for batch in loader:
    toks = minilm_tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = minilm(**toks.to('mps'))
        result = F.normalize(mean_pooling(model_output, toks['attention_mask']), p=2, dim=1)
        embeddings.append(result.cpu())
```


```python
embeddings = torch.cat(embeddings)
```


```python
neural_kmeans = KMeans(n_clusters=4, n_init=20)
neural_kmeans.fit(embeddings)
```

Once again, we can look at the descriptions of the most central documents.


```python
centers = most_central(embeddings, neural_kmeans)
central_texts = [str(np_text[neural_kmeans.labels_ == i][a]) for i, a in enumerate(centers)]
```


```python
summarize(central_texts)
```




    ['The major themes in this documents are  and the power of the gun. The gun was used to kill 25 people, including 25 children. All of us are responsible. The question is not whether, but how. The FBI, the BATF, Ms. Reno, the Prez, and EVERYBODY ELSE in this. world is responsible.',
     'The major themes in this documents are xionist, anti-atheist, and anti-Mormon. The article was written by Rick Anderson, a member of the Church of Jesus Christ of Latter-day Saints. Anderson: "Of all the "preachers" of "truth" on this net, you have struck me as a self-righteous member of a wrecking crew"',
     'The major themes in this documents are atation, space exploration, and life sciences. The author argues that SSF needs to be redesigned to focus on the 3 main functions: microgravity/vacuum process research, life sciences research (adaptation to space) and spacecraft maintenence.',
     'The major themes in this documents are i graphics. This program can let you READ, WRITE and DISPLAY images with different formats. It also let you do some special effects(ROTATION, DITHERING ....) on image. There is no warranty. The author is not responsible for any damage caused by this program.']



Space, graphics, religion and ... guns? Not quite what I expected. 

This notebook will continue to grow as I try other aproaches to the problem. 

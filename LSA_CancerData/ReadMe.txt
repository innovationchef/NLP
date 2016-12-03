This code takes in 200 text files of abstracts of cancer data scraped from scopus. 

It applies Latent Semantic Analysis on this to search for query keywords.

The Term Doc matrix has been filled by frequency on terms in LSA_TermFreq.py

I tested it for the words "HPV" and "Treatment" but the best cosine relatedness I got was 0.359 which I guess is poor 

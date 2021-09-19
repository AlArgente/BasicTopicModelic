# BasicTopicModelic

Basic topic modeling using Latent Dirichlet Allocation (LDA) in Python.

In this repository I will usa LDA for topic modeling. I will use it over the 20newsgroup dataset from sklearn, which contains 20 targets. 

Before applying LDA, it's important to prepare the data, cleaning and preprocesing it. In the Notebook I compare two models, one without cleaning emails or puntuation, and other with full cleaning. In both case I lemmatize the data and not stem it. 

Also I try to explore the correct number of topics that fits better with the problem.

The topic distribution across the documents need to be fixed to work as it's supposed.

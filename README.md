# A Recommender Engine for headlines articles based on headlines embedding.
The project develop and application that suggest to reader more similar articles to the those they already read. It uses the embedding algorithms of headlines to create their own numerical representation, which allows to compute similarity between headlines and get the most similar ones.

For purpose of simplicity, we was satisfied with only headlines that concernes the year of 2018.

# Steps of the project
We build the function "general_process" saved in the preprocessing.py file, to prepare the text data. Its output is the processed_data csv file, that contains the headlines after the preprocessing.

three algorithms are used to build a numerical representation of each headline, We talk about:
 - NMF and LDA factorization: We create a sparse matrix that composed of rows that represent each headlines and columns that represent each word in the entire vocabulary.
 - word2vec : A deeplearning approach, that uses an average word2vec of words composing that headline.
 


# A Recommender Engine for headlines articles using embedded words.
The project develop and application that suggest to reader more similar articles to the those they already read. It uses the embedding algorithms of headlines to create their own numerical representation, which allows to compute similarity between headlines and get the most similar ones.

For purpose of simplicity, we was satisfied only with headlines that concernes the year of 2018.

# Steps of the project
We build the function "general_process" saved in the preprocessing.py file, to prepare the text data. Its output is the processed_data csv file, that contains the headlines after the preprocessing.

three algorithms are used to build a numerical representation of each headline, We talk about:
 - NMF and LDA factorization: We create a sparse matrix that composed of rows that represent each headlines and columns that represent each word in the entire vocabulary.
 - word2vec : A deeplearning approach, that uses an average word2vec of words composing that headline.
those algorithms are exploited with the function "recommender_engine" developed in the recommender py file.

# To excecute the app
Clone the repository in the commend line using the link : https://github.com/akhsassoualid/Headline_Recommender.
```
git clone https://github.com/akhsassoualid/Headline_Recommender.git

```

Install the necessary requirements : 
```
pip install -r requirements.txt

```
Run the application savec in the app.py file

```
streamlit run app.py

```

# Illustrate the application
A simple illustration of the App : 
![Alt text](recom_app.gif)

# Special Thanks:
* Google team of researchers for the [Word2Vec](https://github.com/tmikolov/word2vec) trained model.
* To the team of [Streamlit](https://github.com/streamlit) for their open-source Python library to build applications.
* To [vikashrajluhaniwal](https://medium.com/@vikashrajluhaniwal) for his tutorial about recommendation system.
* To my friends [Rachid](https://github.com/rachideffghi) and [Salih](https://github.com/salihbout) for their help.
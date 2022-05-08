from flask import Flask, Response, render_template, request, url_for, redirect
import json
from wtforms import StringField, Form
import pickle
import regex as re
import numpy as np
import pandas as pd
import surprise
from scipy.sparse import hstack
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

application = Flask(__name__)
 
class SearchForm(Form):
    autocomp = StringField('Enter User Id', id='user_id_autocomplete')

class AnimeSearchForm(Form):
    autocomp = StringField('Search Anime Id', id='anime_title_autocomplete')

# initializing global variables

# loading anime type vectorizer
a_file = open("type_vectorizer.pkl", "rb")
type_vectorizer = pickle.load(a_file)
a_file.close()

# loading anime source vectorizer
a_file = open("source_vectorizer.pkl", "rb")
source_vectorizer = pickle.load(a_file)
a_file.close()

# loading anime studio vectorizer
a_file = open("studio_vectorizer.pkl", "rb")
studio_vectorizer = pickle.load(a_file)
a_file.close()

# loading anime genre vectorizer
a_file = open("genre_vectorizer.pkl", "rb")
genre_vectorizer = pickle.load(a_file)
a_file.close()

# loading anime profile matrix
sample_encoded = sparse.load_npz("sample_encoded.npz")

# loading anime id list
sample_anime_id = np.load('sample_anime_id.npy')

# loading train user id list
a_file = open("train_user_id.pkl", "rb")
train_user_id = pickle.load(a_file)
a_file.close()

# loading train user id list
a_file = open("new_test_anime.pkl", "rb")
new_test_anime = pickle.load(a_file)
a_file.close()

# loading user watched anime dictonary
a_file = open("user_watched_anime_dict.pkl", "rb")
user_watched_anime_dict = pickle.load(a_file)
a_file.close()

# loading sample anime profile dataframe
df_sample_anime_profile = pd.read_csv("df_sample_anime_profile.csv")
df_sample_anime_profile = df_sample_anime_profile.drop(["Unnamed: 0"], axis = 1)

# loading user rated anime dictonary
a_file = open("user_rated_anime_dict.pkl", "rb")
user_rated_anime_dict = pickle.load(a_file)
a_file.close()

# loading anime title dictonary
a_file = open("anime_title_dict.pkl", "rb")
anime_title_dict = pickle.load(a_file)
a_file.close()

# loading anime image url dictonary
a_file = open("anime_with_image_url_dict.pkl", "rb")
anime_with_image_url_dict = pickle.load(a_file)
a_file.close()

 
@application.route('/autocomplete', methods=['GET'])
def autocomplete():
    a_file = open("train_user_id.pkl", "rb")
    train_user_id = pickle.load(a_file)
    a_file.close()   
    train_user_id = list(np.array(train_user_id).astype(str))
    return Response(json.dumps(train_user_id), mimetype='application/json')

@application.route('/anime_autocomplete', methods=['GET'])
def anime_autocomplete():
    a_file = open("anime_title_dict.pkl", "rb")
    anime_title_dict = pickle.load(a_file)
    a_file.close()   
    anime_title_list = list(anime_title_dict.values())
    return Response(json.dumps(anime_title_list), mimetype='application/json')
 
@application.route('/', methods=['GET', 'POST'])
def index(): 
    form = SearchForm(request.form)
    print(request.form)
    if request.method == 'POST':
        if request.form['submit_button'] == "Create Custom User":
            return redirect(url_for("CustomUser"))
        else:
            val = request.form['anime']
            print(id)
            return redirect(url_for("predict", id=val))
    else: 
        return render_template('index.html', form=form)


@application.route('/CustomUser', methods=['GET', 'POST'])
def CustomUser(): 
    form = AnimeSearchForm(request.form)
    return render_template('CustomUser.html', form=form)


def final(df, no_predictions = 10):

    # using CountVectorizer for different anime features
    single_user_type_enc = type_vectorizer.transform(df['type'].values)
    single_user_source_enc = source_vectorizer.transform(df['source'].values)
    single_user_studio_enc = studio_vectorizer.transform(df['studio'].values)
    single_user_genre_enc = genre_vectorizer.transform(df['genre'].values)

    # merging all encoded anime features matrices
    single_user_encoded = hstack((single_user_type_enc, single_user_source_enc, single_user_studio_enc, single_user_genre_enc)).tocsr()

    # creating user profile vector
    user_rating = df['my_score'].values
    user_vec = np.zeros(429)
    for ind,vec in enumerate(single_user_encoded):
        # adding all the anime profile for a particular user by multiplying it with given user rating
        user_vec += vec.toarray()[0]*int(user_rating[ind]) 

    # computing cosine similarity between user profile and anime profile
    user_vec_normalize = normalize(user_vec.reshape(1,-1), norm = 'l2')
    similarity_vec = cosine_similarity(user_vec_normalize, sample_encoded)[0]
    scaler = MinMaxScaler(feature_range=(1, 10))
    content_based_user_ratings = scaler.fit_transform(similarity_vec.reshape(-1, 1)).ravel()

    user_id = df['user_id'].values[0]
    
    # computing hybrid recommender system predicted ratings
    hybrid_user_ratings = []
    if user_id in train_user_id:
        # loading trained knn baseline model
        collaborative_filtering_user_ratings=np.load("knn_baseline_predict/knn_baseline_predict_"+str(user_id)+".npy")
        for i in range(len(content_based_user_ratings)):
            if i in new_test_anime:
                hybrid_user_ratings.append(collaborative_filtering_user_ratings[i])
            else:
                val = content_based_user_ratings[i]*0.03 + collaborative_filtering_user_ratings[i]
                hybrid_user_ratings.append(val)
    else: # for new custom users 
        for i in range(len(content_based_user_ratings)):
            val = content_based_user_ratings[i]
            hybrid_user_ratings.append(val)
    
    hybrid_user_rating_sorted_index =  np.array(hybrid_user_ratings).argsort()[::-1][1:]
    user_watched_anime_id = df['anime_id'].values
    recommended_anime_id = []
    for i in hybrid_user_rating_sorted_index:
        if sample_anime_id[i] not in user_watched_anime_id:
            recommended_anime_id.append(sample_anime_id[i])
        if len(recommended_anime_id) ==  no_predictions:
            break

    return recommended_anime_id


@application.route('/<id>', methods=['GET'])
def predict(id):

    # getting list of watched anime id for particular user id
    watched_anime_list = user_watched_anime_dict[int(id)]

    # creating dataframe for particular user id
    df = pd.DataFrame()
    for i in watched_anime_list:
        df = df.append(df_sample_anime_profile[df_sample_anime_profile['anime_id']==i])
    
    # getting list of ratings given particular user id and storing it in 'my_score' column
    df['my_score'] = user_rated_anime_dict[int(id)]
    df['user_id'] = [int(id)]*len(watched_anime_list)
    
    # passing df dataframe in final function to get recommended anime id 
    recommended_anime_id = final(df)

    # creating recommended anime dictonary to pass it in 'recommended' html page
    anime_dict = dict()
    for id in recommended_anime_id:
        anime_dict[id] = [int(id) ,anime_title_dict[int(id)], anime_with_image_url_dict[int(id)]]

    # creating watched anime dictonary to pass it in 'recommended' html page
    df = df.sort_values(by=['my_score'], ascending=False)
    watched_anime_list = df['anime_id'].values
    watched_anime_dict =  dict()
    for id in watched_anime_list:
        watched_anime_dict[id] = [int(id) ,anime_title_dict[int(id)], anime_with_image_url_dict[int(id)], int(df[df['anime_id']==id]['my_score'].values)]

    return render_template('recommended.html', anime_dict=anime_dict, watched_anime_dict=watched_anime_dict)


@application.route('/custom_user_predict', methods=['POST'])
def custom_user_predict():

    # reading input of custom user and storing anime title with its given rating in a dictonary
    custom_user_ratings = request.form['review_text']
    split_custom_user_ratings = custom_user_ratings.split("|")
    anime_rating_dict = dict()
    for i in split_custom_user_ratings:
        title_and_ratings = i.split(":")
        anime_rating = int(title_and_ratings[-1])
        if len(title_and_ratings) == 2:
            anime_title = re.sub(r"^\s+|\s+$", "", title_and_ratings[0])
        else:
            for ind,val in enumerate(title_and_ratings[:-1]):
                if ind == 0:
                    anime_title = val
                else:
                    anime_title += ":" + val
            anime_title = re.sub(r"^\s+|\s+$", "", anime_title)
        anime_rating_dict[anime_title] = anime_rating

    # creating dataframe for new custom user
    custom_df = pd.DataFrame(anime_rating_dict.items(), columns = ['title', 'my_score'])
    custom_df = pd.merge(custom_df, df_sample_anime_profile, on = 'title')
    custom_df['user_id'] = [int(1)]*len(anime_rating_dict) # assigning unique user id 

    # passing df dataframe in final function to get recommended anime id 
    recommended_anime_id = final(custom_df)

    # creating recommended anime dictonary to pass it in 'recommended' html page
    anime_dict = dict()
    for id in recommended_anime_id:
        anime_dict[id] = [int(id) ,anime_title_dict[int(id)], anime_with_image_url_dict[int(id)]]

    # creating watched anime dictonary to pass it in 'recommended' html page
    custom_df = custom_df.sort_values(by=['my_score'], ascending=False)   
    watched_anime_list = custom_df['anime_id'].values
    watched_anime_dict =  dict()
    for id in watched_anime_list:
        watched_anime_dict[id] = [int(id) ,anime_title_dict[int(id)], anime_with_image_url_dict[int(id)], int(custom_df[custom_df['anime_id']==id]['my_score'].values)]

    return render_template('recommended.html', anime_dict=anime_dict, watched_anime_dict=watched_anime_dict)

if __name__ == '__main__':
    application.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, jsonify
from data.reddit_api_data import RedditApiData
import os
import ast
from model.model import predict_model
from scripts.Preprocessing_full import prediction_preprocessing, prediction_vectorize
import nltk
import numpy
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

CLIENT_ID = os.environ["CLIENT_ID"]
SECRET_KEY = os.environ["SECRET_KEY"]
REDDIT_USERNAME = os.environ["REDDIT_USERNAME"]

app = Flask(__name__)

reddit_api_data = RedditApiData(CLIENT_ID, SECRET_KEY, REDDIT_USERNAME)

@app.route('/', methods=['GET', 'POST'])
def index():
    reddit_user_info = None
    reddit_comments = None
    # with open(f'model/model3.pkl', 'rb') as file:
    #     model = pickle.load(file)


    if request.method == 'POST':
        field_content = request.form.get('content') # get content from form

        try:
            reddit_user_info = reddit_api_data.get_user_infos(field_content) # get user info from Reddit API
            print(type(field_content))
            reddit_comments = reddit_api_data.get_comments(field_content) # get user comments from Reddit API
            print("Api data collected")
            df_pred_pp = prediction_preprocessing(reddit_comments) # preprocess the comments
            print("df_pred_pp loaded")
            df_pred_list = prediction_vectorize(df_pred_pp) # vectorize the comments
            print("df_pred_list loaded")
            prediction = predict_model(texts = df_pred_list, verbose = False)
            prediction = prediction['type_prediction']

            if reddit_user_info is not None:
                # Redirect to the result page with the reddit_user_info as a parameter
                return redirect(url_for('result', user_info=reddit_user_info,reddit_comments=reddit_comments, prediction=prediction))

        except Exception as e:
            # Handle the exception as per your requirement
            print(f"An error occurred: {str(e)}")

    return render_template('index.html', reddit_user_info=reddit_user_info)

@app.route('/result', methods=['GET'])
def result():
    user_info_encoded = request.args.get('user_info')
    user_type = request.args.get('prediction')
    user_info = ast.literal_eval(user_info_encoded)
    # Create a rando type to test the template
    # type = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']
    # user_type = random.choice(type)
    template_path = f'type/{user_type}.html'


    return render_template(template_path, reddit_user_info=user_info)

if __name__ == '__main__':
    app.run(port=8080, debug=True)

from urllib.parse import urlparse
from flask import Flask, request, render_template, send_from_directory
import pickle
import numpy as np
from feature import FeatureExtraction
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('pickle/model.pkl', 'rb'))

# Load models
cnn = pickle.load(
    open('models/CNN_Oversampling_df_preprocessed_1.5.2.pkl', 'rb'))
rf = pickle.load(
    open('models\RF_OVERSAMPLING_df_preprocessed_model.pkl', 'rb'))
# cnn = joblib.load('models/CNN_Oversampling_df_preprocessed.joblib')
# rf = joblib.load('models/RF_OVERSAMPLING_df_preprocessed_model.joblib')

# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(
    1, 3), stop_words='english', min_df=1)


def clean_input(user_input):
    return user_input.strip().replace('"', '').replace("'", '')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_text(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag) or wordnet.NOUN)
        for token, tag in tagged_tokens
    ]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def normalize(text, column):
    if column != 'location':
        text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    text = ' '.join(words)
    return text


def preprocess_input(data):
    for column in data.columns:
        data[column] = data[column].apply(clean_input)
        data[column] = data[column].apply(
            lambda x: None if isinstance(x, str) and not x.strip() else x)
        data[column] = data[column].apply(
            lambda x: normalize(x, column) if x else x)
        data[column] = data[column].apply(
            lambda x: lemmatize_text(x) if x else x)

    data.dropna(axis=1, how='all', inplace=True)

    return data


def calculate_weights(tfidf_features):
    weight_cnn_1 = np.mean(tfidf_features) * 0.3
    weight_cnn_2 = np.std(tfidf_features) * 0.2
    weight_rf_1 = np.max(tfidf_features) * 0.4
    weight_rf_2 = np.min(tfidf_features) * 0.1
    return {
        'cnn_per_1': weight_cnn_1,
        'cnn_per_2': weight_cnn_2,
        'rf_per_1': weight_rf_1,
        'rf_per_2': weight_rf_2
    }


def predict_advertisement(data):
    preprocessed_data = preprocess_input(data)

    df_preprocessed_columns = preprocessed_data.columns.tolist()
    all_features = []

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(
        1, 3), stop_words='english', min_df=1)

    for column in df_preprocessed_columns:
        column_data = preprocessed_data[column]
        token_strings = [str(token) for token in column_data]

        if not any(token_strings) or all(not token.strip() for token in token_strings):
            continue

        try:
            text_features = vectorizer.fit_transform(token_strings)
        except ValueError:
            continue

        my_array = text_features.toarray()
        features = pd.DataFrame(
            my_array, columns=vectorizer.get_feature_names_out())
        all_features.append(features)

    if all_features:
        df_preprocessed_features = pd.concat(all_features, axis=1)
    else:
        df_preprocessed_features = pd.DataFrame()

    if not df_preprocessed_features.empty:
        new_features4 = pd.DataFrame(np.zeros((1, 13275)))
        num_cols = df_preprocessed_features.shape[1]
        new_features4.iloc[:, :num_cols] = df_preprocessed_features.values

        weights = calculate_weights(df_preprocessed_features.values)
        num_samples = df_preprocessed_features.shape[0]

        cnn_per = cnn.predict(new_features4)
        rf_per = rf.predict(new_features4)

        cnn_per_tiled = np.tile(cnn_per, (num_samples, 1))
        rf_per_tiled = np.tile(rf_per, (num_samples, 1))

        blended_new_pred = (
            weights['cnn_per_1'] * cnn_per_tiled +
            weights['cnn_per_2'] * cnn_per_tiled +
            weights['rf_per_1'] * rf_per_tiled +
            weights['rf_per_2'] * rf_per_tiled
        )

        blended_new_pred /= (weights['cnn_per_1'] + weights['cnn_per_2'] +
                             weights['rf_per_1'] + weights['rf_per_2'])
        print(blended_new_pred)
        blended_new_pred_rounded = np.round(blended_new_pred, 4)
        print(blended_new_pred_rounded)
        threshold = 0.8539
        prediction = np.where(blended_new_pred_rounded < threshold, 1, 0)
        print(prediction)

        if prediction == 0:
            return "The advertisement is Real"
        else:
            return "The advertisement is Fake"
    else:
        return 'Insufficient Data'


def is_whitelisted(url):
    whitelist = [
        "google.com", "bing.com", "yahoo.com",
        "facebook.com", "twitter.com", "instagram.com",
        "gmail.com", "outlook.com", "amazon.com",
        "ebay.com", "etsy.com", "walmart.com",
        "chase.com", "bankofamerica.com", "paypal.com",
        "cnn.com", "bbc.co.uk", "nytimes.com",
        "harvard.edu", "mit.edu", "coursera.org",
        "usa.gov", "nasa.gov",
        "edu.sa", "edu.sa",  # Educational institutions in Saudi Arabia
        "gov.sa"    # Government domains
    ]
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    try:
        # Parse the URL to get the hostname
        parsed_url = urlparse(url)
        host = parsed_url.netloc.lower().replace("www.", "").strip('/')

        # Check if the host or its parent domain is in the whitelist
        if any(host.endswith(domain) for domain in whitelist):
            return 1
        else:
            return -1
    except Exception as e:
        print("Error parsing URL:", e)
        return -1


@app.route('/enter-details', methods=['GET', 'POST'])
def enter_details():
    if request.method == 'POST':
        # Extract job details from form
        job_title = clean_text(request.form['jobTitle'])
        company_name = clean_text(request.form['companyName'])
        location = clean_text(request.form['location'])
        job_description = clean_text(request.form['jobDescription'])
        job_function = clean_text(request.form.get('function', ''))
        requirements = clean_text(request.form.get('requirements', ''))
        education = clean_text(request.form.get('education', ''))
        benefits = clean_text(request.form.get('benefit', ''))
        other_info = clean_text(request.form.get('other', ''))
        url = request.form.get('url', '')
        url_result = request.form.get('url_result', '')
        uid = request.form.get('uid', '')

        # Create a DataFrame from the form data
        job_data = pd.DataFrame([{
            'jobTitle': job_title,
            'companyName': company_name,
            'location': location,
            'jobDescription': job_description,
            'function': job_function,
            'requirements': requirements,
            'education': education,
            'benefits': benefits,
            'otherInfo': other_info
        }])

        # get first row in job data as a dictionary
        job_dict = {
            'job_title': job_title,
            'company_name': company_name,
            'location': location,
            'job_description': job_description,
            'job_function': job_function,
            'requirements': requirements,
            'education': education,
            'benefits': benefits,
            'other_info': other_info,
            'url': url,
            'url_result': url_result,
            'uid': uid,
        }

        # Process the job ad details
        prediction_result = predict_advertisement(job_data)

        return render_template('result.html', result=prediction_result, job_dict=job_dict)

    return render_template('enter-details.html')


# Route for login
@app.route('/login')
def login():
    return render_template('login.html')

# Route for reset


@app.route('/reset')
def reset():
    return render_template('reset.html')
# Route for signup


@app.route('/signup')
def signup():
    return render_template('Signup.html')
# Custom route to serve images


@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory('images', filename)

# Route for serving CSS files


@app.route('/css/<path:filename>')
def css(filename):
    return send_from_directory('css', filename)

# Route for serving JS files


@app.route('/js/<path:filename>')
def js(filename):
    return send_from_directory('js', filename)

# Route for serving fonts


@app.route('/fonts/<path:filename>')
def fonts(filename):
    return send_from_directory('fonts', filename)


@app.route('/profile')
def profile():
    return render_template('Profile.html')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    return render_template('History.html')


@app.route('/verify', methods=['GET', 'POST'])
def verify_page():
    result = None
    entered_url = None
    history = None
    if request.method == 'POST':
        entered_url = request.form.get('url')
        uid = request.form.get('uid')
        obj = FeatureExtraction(entered_url)
        features = np.array(obj.getFeaturesList()).reshape(1, -1)

        # Check whitelist separately
        whitelist_result = is_whitelisted(entered_url)

        if whitelist_result == 1:
            result = "Safe website"
        else:
            # Predict with the model
            prediction = model.predict(features)[0]
            result = "Safe website" if prediction == 1 else "Suspicious website"
        # todo: save history
        history = {
            'url': entered_url,
            'result': result,
            'uid': uid,
        }
    return render_template('job-advertisement.html', result=result, entered_url=entered_url, history=history)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    print(text)
    return text


if __name__ == '__main__':
    app.run(debug=True)

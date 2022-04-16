import numpy as np
import pandas as pd
import math
from tqdm.notebook import trange, tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

TF_QUANTITY = 100

def get_vocabulary(df):
    count_vectorizer = CountVectorizer(lowercase=False, stop_words=[])
    cv_fit = count_vectorizer.fit_transform(df['content'])
    word_list = count_vectorizer.get_feature_names()
    frequecy_array = cv_fit.toarray()
    count_list = frequecy_array.sum(axis=0)
    vocabulary = (dict(zip(word_list, count_list)))
    return vocabulary, frequecy_array, word_list


def get_bigrams(df):
    count_vectorizer = CountVectorizer(
        lowercase=False, stop_words=[], ngram_range=(2, 2))
    cv_fit = count_vectorizer.fit_transform(df['content'])
    word_list = count_vectorizer.get_feature_names()
    frequecy_array = cv_fit.toarray()
    count_list = frequecy_array.sum(axis=0)
    vocabulary = (dict(zip(word_list, count_list)))
    return vocabulary, frequecy_array, word_list


def get_doc(df, chosen_words):
    return df['content'].apply(lambda y: ' '.join(
        [x for x in y.split() if x in chosen_words]))


def get_bigram_doc(df, chosen_words):
    def select_only_relevant_bigrams(text):
        bigrams_in_text = [b for l in [text]
                           for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        return ' '.join([' '.join(w) for w in bigrams_in_text if ' '.join(w) in chosen_words])
    return df['content'].apply(select_only_relevant_bigrams)


def get_relevant_words(df):
    return list(df.sort_values(
        by=['diff'], ascending=False)['term'])

def select_shared_terms(voc1, voc2):
    # select words that are in bolth vocabularies
    list_sexist_sorted_terms = []
    for key, value in sorted(voc1.items(), key=lambda item: item[1]):
        list_sexist_sorted_terms.append(key)

    shared = []

    for word in list_sexist_sorted_terms:
        if word in voc2.keys():
            shared.append(word) 
    return shared

def select_word_freq(shared, sexist, not_sexist):
    freq = {
        'term': [],
        'sexist-freq': [],
        'not-sexist-freq': [],
        'diff': []
    }
    for term in shared:
        freq['term'].append(term)
        if term in sexist.keys():
            freq['sexist-freq'].append(sexist[term])
        else:
            freq['sexist-freq'].append(0)

        if term in not_sexist.keys():
            freq['not-sexist-freq'].append(not_sexist[term])
        else:
            freq['not-sexist-freq'].append(0)

    freq['diff'] = sexist[term] - not_sexist[term]
    freq = pd.DataFrame(freq)

    # normalizind frequencies
    sum_sexist = sum(freq['sexist-freq'])
    freq['sexist-freq'] = freq['sexist-freq'].apply(
        lambda x: x/sum_sexist)
    sum_not_sexist = sum(freq['not-sexist-freq'])
    freq['not-sexist-freq'] = freq['not-sexist-freq'].apply(
        lambda x: x/sum_not_sexist)
    freq['diff'] = freq['sexist-freq'] - \
        freq['not-sexist-freq']

    sexist_terms = freq[freq['diff'] > 0]
    not_sexist_terms = freq[freq['diff'] < 0]

    # most relevant terms to sexist comments
    sexist_ = sexist_terms.sort_values(
        by='diff', ascending=False)

    # most relevant terms to not sexist comments
    not_sexist_ = not_sexist_terms.sort_values(
        by='diff', ascending=True)

    return sexist_, not_sexist_

def generate_cross_validation_train_and_test(df):
    train_size = math.floor(df.shape[0] * 0.9)
    test_size = math.ceil(df.shape[0] * 0.1)
    for i in range (0,10):
        dataframe = df.sample(frac=1)
        train = dataframe.iloc[:train_size]
        test = dataframe.iloc[train_size:]

        # ------ select most relevant unigrams and bigrams to sexist and not sexist context
        sexist_comments = dataframe[dataframe['avg'] > 0.5]
        sexist_vocabulary, sexist_frequency_array, sexist_word_list = get_vocabulary(sexist_comments)
        sexist_bigrams, sexist_frequency_array, sexist_word_list = get_bigrams(sexist_comments)

        not_sexist_comments = dataframe[dataframe['avg'] < 0.5]    
        not_sexist_vocabulary, not_sexist_frequency_array, not_sexist_word_list = get_vocabulary(not_sexist_comments)
        not_sexist_bigrams, not_sexist_frequency_array, not_sexist_word_list = get_bigrams(not_sexist_comments)

        shared_unigrams = select_shared_terms(sexist_vocabulary, not_sexist_vocabulary)
        sexist_unigrams, not_sexist_unigrams = select_word_freq(shared_unigrams, sexist_vocabulary, not_sexist_vocabulary)

        shared_bigrams = select_shared_terms(sexist_bigrams, not_sexist_bigrams)
        sexist_bigrams, not_sexist_bigrams = select_word_freq(shared_bigrams, sexist_bigrams, not_sexist_bigrams)

        # ------ calculate and serialize term frequency to sexist and not sexist unigrams and bigrams
        sexist_vectorizer = TfidfVectorizer(
            stop_words=[],
            use_idf=False,
            norm=None,
            decode_error='replace',
            max_features=TF_QUANTITY,
        )
        not_sexist_vectorizer = TfidfVectorizer(
            stop_words=[],
            use_idf=False,
            decode_error='replace',
            max_features=TF_QUANTITY,
        )
        sexist_bigrams_vectorizer = TfidfVectorizer(
            stop_words=[],
            use_idf=False,
            ngram_range=(2, 2),
            decode_error='replace',
            max_features=TF_QUANTITY,
        )
        not_sexist_bigrams_vectorizer = TfidfVectorizer(
            stop_words=[],
            use_idf=False,
            ngram_range=(2, 2),
            decode_error='replace',
            max_features=TF_QUANTITY,
        )

        relevant_sexist_words = get_relevant_words(sexist_unigrams)
        relevant_not_sexist_words = get_relevant_words(
            not_sexist_unigrams)

        relevant_sexist_bigrams = get_relevant_words(sexist_bigrams)
        relevant_not_sexist_bigrams = get_relevant_words(
            not_sexist_bigrams)

        # ------ sexist unigrams TF
        sexist_doc = get_doc(sexist_comments, relevant_sexist_words)
        not_sexist_doc = get_doc(not_sexist_comments, relevant_sexist_words)
        sexist_tf = pd.DataFrame(sexist_vectorizer.fit_transform(sexist_doc).toarray())
        not_sexist_tf = pd.DataFrame(not_sexist_vectorizer.fit_transform(not_sexist_doc).toarray())
        tf_sexist_dataframe = pd.concat([sexist_tf, not_sexist_tf]).fillna(0)
        tf_sexist_dataframe.columns = [f'TFus_{i}' for i in range(100)]

        # ------ sexist bigrams TF
        sexist_doc = get_bigram_doc(sexist_comments, relevant_sexist_bigrams)
        not_sexist_doc = get_bigram_doc(not_sexist_comments, relevant_sexist_bigrams)
        sexist_bigrams_tf = pd.DataFrame(sexist_bigrams_vectorizer.fit_transform(sexist_doc).toarray())
        not_sexist_bigrams_tf = pd.DataFrame(not_sexist_bigrams_vectorizer.fit_transform(not_sexist_doc).toarray())
        tf_sexist_bigrams_dataframe = pd.concat([sexist_bigrams_tf, not_sexist_bigrams_tf]).fillna(0)
        tf_sexist_bigrams_dataframe.columns = [f'TFbs_{i}' for i in range(100)]

        # ------ not sexist unigrams TF
        sexist_doc = get_doc(sexist_comments, relevant_not_sexist_words)
        not_sexist_doc = get_doc(not_sexist_comments, relevant_not_sexist_words)
        sexist_tf = pd.DataFrame(sexist_vectorizer.fit_transform(sexist_doc).toarray())
        not_sexist_tf = pd.DataFrame(not_sexist_vectorizer.fit_transform(not_sexist_doc).toarray())
        tf_not_sexist_dataframe = pd.concat([sexist_tf, not_sexist_tf]).fillna(0)
        tf_not_sexist_dataframe.columns = [f'TFun_{i}' for i in range(100)]

        # ------ not sexist bigrams TF
        sexist_doc = get_bigram_doc(sexist_comments, relevant_not_sexist_bigrams)
        not_sexist_doc = get_bigram_doc(not_sexist_comments, relevant_not_sexist_bigrams)
        sexist_bigrams_tf = pd.DataFrame(sexist_bigrams_vectorizer.fit_transform(sexist_doc).toarray())
        not_sexist_bigrams_tf = pd.DataFrame(not_sexist_bigrams_vectorizer.fit_transform(not_sexist_doc).toarray())
        tf_not_sexist_bigrams_dataframe = pd.concat([sexist_bigrams_tf, not_sexist_bigrams_tf]).fillna(0)
        tf_not_sexist_bigrams_dataframe.columns = [f'TFbn_{i}' for i in range(100)]

        tf_dataframe = pd.concat([tf_sexist_dataframe,
                                  tf_not_sexist_dataframe,
                                  tf_sexist_bigrams_dataframe,
                                  tf_not_sexist_bigrams_dataframe], axis=1)

        # ------ define quantitative features to train
        likes_df = np.array(pd.concat([sexist_comments['likes'], not_sexist_comments['likes']]).fillna(0))
        dislikes_df = np.array(pd.concat([sexist_comments['dislikes'], not_sexist_comments['dislikes']]).fillna(0))
        char_qty_df = np.array(pd.concat([sexist_comments['char-qty'], not_sexist_comments['char-qty']]).fillna(0))
        word_qty_df = np.array(pd.concat([sexist_comments['word-qty'], not_sexist_comments['word-qty']]).fillna(0))
        sexist_y = sexist_comments['avg'].apply(lambda x: 1)
        not_sexist_y = not_sexist_comments['avg'].apply(lambda x: 0)
        y_df = np.array(pd.concat([sexist_y, not_sexist_y]))

        X_train = tf_dataframe
        X_train['likes'] = likes_df
        X_train['dislikes'] = dislikes_df
        X_train['char-qty'] = char_qty_df
        X_train['word-qty'] = word_qty_df
        X_train['sexist'] = y_df
        X_train = X_train.fillna(0)
        X_train = X_train.sample(frac=1)
        X_train.to_csv(f'./data/{i+1}_train.csv', index=False)

        # define features to test

        doc_vectorizer = TfidfVectorizer(
            stop_words=[],
            use_idf=False,
            norm=None,
            decode_error='replace',
            max_features=TF_QUANTITY,
        )    

        doc = get_doc(test, relevant_sexist_words)
        tf = pd.DataFrame(doc_vectorizer.fit_transform(doc).toarray(), columns=[f'TFus_{i}' for i in range(100)])
        tf_sexist_dataframe = pd.concat([tf]).fillna(0)

        doc = get_bigram_doc(test, relevant_sexist_bigrams)
        tf = pd.DataFrame(doc_vectorizer.fit_transform(doc).toarray(), columns=[f'TFbs_{i}' for i in range(100)])
        tf_sexist_bigrams_dataframe = pd.concat([tf]).fillna(0)

        doc = get_doc(test, relevant_not_sexist_words)
        tf = pd.DataFrame(doc_vectorizer.fit_transform(doc).toarray(), columns=[f'TFun_{i}' for i in range(100)])
        tf_not_sexist_dataframe = pd.concat([tf]).fillna(0)

        doc = get_bigram_doc(test, relevant_not_sexist_bigrams)
        tf = pd.DataFrame(doc_vectorizer.fit_transform(doc).toarray(), columns=[f'TFbn_{i}' for i in range(100)])
        tf_not_sexist_bigrams_dataframe = pd.concat([tf]).fillna(0)

        tf_test_dataframe = pd.concat([tf_sexist_dataframe,
                                      tf_not_sexist_dataframe,
                                      tf_sexist_bigrams_dataframe,
                                      tf_not_sexist_bigrams_dataframe], axis=1)
        X_test = tf_dataframe
        X_test['likes'] = likes_df
        X_test['dislikes'] = dislikes_df
        X_test['char-qty'] = char_qty_df
        X_test['word-qty'] = word_qty_df
        X_test['sexist'] = y_df
        X_test = X_test.fillna(0)
        X_test = X_test.sample(frac=1)
        X_test.to_csv(f'./data/{i+1}_test.csv', index=False)

def select_df_columns(df, columns):
    list_features = [df[df.columns[list(df.columns).index(x)]] for x in columns]
    X = pd.DataFrame(list_features).transpose()
    return X

def build_results_report(title,precision,recall,fscore,f1macro):
    report = {
        'title': title,
        'precision': {'1': [], '0': []},
        'recall': {'1': [], '0': []},
        'f1': {'1': [], '0': []},
        'f1_macro': []
    }
    report['precision']['1'].append(precision[0])
    report['recall']['1'].append(recall[0])
    report['f1']['1'].append(fscore[0])
    report['precision']['0'].append(precision[1])
    report['recall']['0'].append(recall[1])
    report['f1']['0'].append(fscore[1])
    report['f1_macro'].append(f1macro)
    return report

def print_report(report):
    print(f'>>>> {report["title"]} results')
    print('\t\t sexist \t not-sexist')
    print('precision\t %.5f \t %.5f' % (
        np.mean(report["precision"]["1"]), np.mean(report["precision"]["0"])))
    print('recall\t\t %.5f \t %.5f' %
          (np.mean(report["recall"]["1"]), np.mean(report["recall"]["0"])))
    print('f1\t\t %.5f \t %.5f' %
          (np.mean(report["f1"]["1"]), np.mean(report["f1"]["0"])))
    print('')
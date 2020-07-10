{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "import os\n",
    "import gc\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train = pd.read_csv('../input/train.csv')\n",
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Total number of question pairs for training: {}'.format(len(df_train)))\n",
    "print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))\n",
    "qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())\n",
    "print('Total number of questions in the training data: {}'.format(len(\n",
    "    np.unique(qids))))\n",
    "print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import log_loss\n",
    "\n",
    "p = df_train['is_duplicate'].mean() \n",
    "print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))\n",
    "\n",
    "df_test = pd.read_csv('../input/test.csv')\n",
    "sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})\n",
    "sub.to_csv('naive_submission.csv', index=False)\n",
    "sub.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test = pd.read_csv('../input/test.csv')\n",
    "df_test.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Total number of question pairs for testing: {}'.format(len(df_test)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)\n",
    "test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dist_train = train_qs.apply(lambda x: len(x.split(' ')))\n",
    "dist_test = test_qs.apply(lambda x: len(x.split(' ')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from wordcloud import WordCloud\n",
    "cloud = WordCloud(width=1440, height=1080).generate(\" \".join(train_qs.astype(str)))\n",
    "plt.figure(figsize=(20, 15))\n",
    "plt.imshow(cloud)\n",
    "plt.axis('off')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "qmarks = np.mean(train_qs.apply(lambda x: '?' in x))\n",
    "math = np.mean(train_qs.apply(lambda x: '[math]' in x))\n",
    "fullstop = np.mean(train_qs.apply(lambda x: '.' in x))\n",
    "capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))\n",
    "capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))\n",
    "numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))\n",
    "\n",
    "print('Questions with question marks: {:.2f}%'.format(qmarks * 100))\n",
    "print('Questions with [math] tags: {:.2f}%'.format(math * 100))\n",
    "print('Questions with full stops: {:.2f}%'.format(fullstop * 100))\n",
    "print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))\n",
    "print('Questions with capital letters: {:.2f}%'.format(capitals * 100))\n",
    "print('Questions with numbers: {:.2f}%'.format(numbers * 100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords\n",
    "\n",
    "stops = set(stopwords.words(\"english\"))\n",
    "\n",
    "def word_match_share(row):\n",
    "    q1words = {}\n",
    "    q2words = {}\n",
    "    for word in str(row['question1']).lower().split():\n",
    "        if word not in stops:\n",
    "            q1words[word] = 1\n",
    "    for word in str(row['question2']).lower().split():\n",
    "        if word not in stops:\n",
    "            q2words[word] = 1\n",
    "    if len(q1words) == 0 or len(q2words) == 0:\n",
    "        return 0\n",
    "    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]\n",
    "    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]\n",
    "    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))\n",
    "    return R"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import Counter\n",
    "\n",
    "def get_weight(count, eps=10000, min_count=2):\n",
    "    if count < min_count:\n",
    "        return 0\n",
    "    else:\n",
    "        return 1 / (count + eps)\n",
    "\n",
    "eps = 5000 \n",
    "words = (\" \".join(train_qs)).lower().split()\n",
    "counts = Counter(words)\n",
    "weights = {word: get_weight(count) for word, count in counts.items()}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tfidf_word_match_share(row):\n",
    "    q1words = {}\n",
    "    q2words = {}\n",
    "    for word in str(row['question1']).lower().split():\n",
    "        if word not in stops:\n",
    "            q1words[word] = 1\n",
    "    for word in str(row['question2']).lower().split():\n",
    "        if word not in stops:\n",
    "            q2words[word] = 1\n",
    "    if len(q1words) == 0 or len(q2words) == 0:\n",
    "        return 0\n",
    "    \n",
    "    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]\n",
    "    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]\n",
    "    \n",
    "    R = np.sum(shared_weights) / np.sum(total_weights)\n",
    "    return R"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_auc_score\n",
    "print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))\n",
    "print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = pd.DataFrame()\n",
    "x_test = pd.DataFrame()\n",
    "x_train['word_match'] = train_word_match\n",
    "x_train['tfidf_word_match'] = tfidf_train_word_match\n",
    "x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)\n",
    "x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)\n",
    "\n",
    "y_train = df_train['is_duplicate'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pos_train = x_train[y_train == 1]\n",
    "neg_train = x_train[y_train == 0]\n",
    "\n",
    "p = 0.165\n",
    "scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1\n",
    "while scale > 1:\n",
    "    neg_train = pd.concat([neg_train, neg_train])\n",
    "    scale -=1\n",
    "neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])\n",
    "print(len(pos_train) / (len(pos_train) + len(neg_train)))\n",
    "\n",
    "x_train = pd.concat([pos_train, neg_train])\n",
    "y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()\n",
    "del pos_train, neg_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.cross_validation import train_test_split\n",
    "\n",
    "x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import xgboost as xgb\n",
    "\n",
    "# Set our parameters for xgboost\n",
    "params = {}\n",
    "params['objective'] = 'binary:logistic'\n",
    "params['eval_metric'] = 'logloss'\n",
    "params['eta'] = 0.02\n",
    "params['max_depth'] = 4\n",
    "\n",
    "d_train = xgb.DMatrix(x_train, label=y_train)\n",
    "d_valid = xgb.DMatrix(x_valid, label=y_valid)\n",
    "\n",
    "watchlist = [(d_train, 'train'), (d_valid, 'valid')]\n",
    "\n",
    "bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "d_test = xgb.DMatrix(x_test)\n",
    "p_test = bst.predict(d_test)\n",
    "\n",
    "sub = pd.DataFrame()\n",
    "sub['test_id'] = df_test['test_id']\n",
    "sub['is_duplicate'] = p_test\n",
    "sub.to_csv('simple_xgb.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

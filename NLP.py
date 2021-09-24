# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:23:40 2021
@author: 119987
"""

"""
Strcuture
https://jsiwu94.github.io/Airlines-on-Twitter/
In this project, I worked with a group of 5 for our last final project in my Master’s degree. With this, we aimed to understand customer complaints on social media, specifically Twitter, regarding the major airlines in the US. To learn more about our project and the codes we used to implement it, please check out our github repository here. This article was also featured by AnalyticsVidhya on Medium.com.
Objective
Customer sentiment:
To identify positive and negative opinions, emotions, and evaluations.
Identify negative topics:
Derive negative topics that people are likely to mention when talking with their experience with airlines.
Derive actionable insights:
Insights may later be used by airlines in planning and execution of customer service initiatives, media relationships etc.

Strcuture
https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184

In the next parts of this series we’ll look at more sophisticated ways to get better performance out of our classifier.
Text Processing: Stemming/Lemmatizing to convert different forms of each word into one.
n-grams: Instead of just single-word tokens (1-gram/unigram) we can also include word pairs.
Representations: Instead of simple, binary vectors we can use word counts or TF-IDF to transform those counts.
Algorithms: In addition to Logistic Regression, we’ll see how Support Vector Machines perform.



Intro how does sentiment works (with img)
https://monkeylearn.com/blog/text-mining-sentiment-analysis/


Ref from Will
https://www.cnblogs.com/chen8023miss/p/11458571.html
https://blog.csdn.net/weixin_43579079/article/details/99324579
"""









"""
# Data collection
"""

import pandas as pd
mgm_macau_reviews = pd.read_csv('Z:/MGM/My work/Python/TripReviews_edited.csv', encoding='utf-8')

mgm_cotai_reviews = pd.read_csv('Z:/MGM/My work/Python/TripReviews_mgm_cotai_edited.csv', encoding='utf-8')




"""
# Data cleansing
"""


#mgm_macau_reviews = pd.read_csv('C:/Users/Zing/OneDrive/GitHub/Python/NLP/TripReviews_edited.csv'#, index_col=0
#                                    )

# Show dataframe
print(mgm_macau_reviews)

mgm_macau_reviews.dropna(how='all', inplace = True) 

print(mgm_macau_reviews['username'][0])
print(mgm_macau_reviews['review'][0])



mgm_cotai_reviews.dropna(how='all', inplace = True) 
print(mgm_cotai_reviews['username'][0])
print(mgm_cotai_reviews['review'][0])


 
 
#https://medium.com/@ODSC/creating-if-elseif-else-variables-in-python-pandas-7900f512f0e4
#df['sentiment'] = df.apply(lambda row: row.Score1 + row.Score2, axis = 1) 



mgm_macau_reviews['sentiment'] = mgm_macau_reviews.star.map(\
lambda x: 'Positive' if 2<x else \
'Negative' )

mgm_cotai_reviews['sentiment'] = mgm_cotai_reviews.star.map(\
lambda x: 'Positive' if 2<x else \
'Negative' )   

    
    
"""
# Data exploration
"""

import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud#, STOPWORDS
from stop_words import safe_get_stop_words
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('names')
#nltk.download([
#"names",
#"stopwords",
#"state_union",
#"twitter_samples",
#"movie_reviews",
#"averaged_perceptron_tagger",
#"vader_lexicon",
#"punkt",
#])

sentiment = 'Positive'


# Combine all reviews for the desired sentiment
combined_text = " ".join([review for review in mgm_macau_reviews['review'][mgm_macau_reviews['sentiment']==sentiment]])


#stopwords = nltk.corpus.stopwords.words('english')
STOPWORDS = stopwords.words('english')
newStopWords = ['hotel','mgm','macau','macau','room','staff','hotels','friend','free','us','check'
                                     ,'much','visit' , 'many' , 'check','even','way','bit','arrived'
                                     ,'night','casino','service','stayed','got','around','provide','really'
                                     ,'every','even','make','check','took','provided','wynn','staying','booked'
                                     ,'Venetian','trip','made','will', 'went','always','say','came','need'
                                     ,'day','time','u','one','first'
                                     "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"
                                     ]
STOPWORDS.extend(newStopWords)


#width=800, height=400
#https://stackoverflow.com/questions/28786534/increase-resolution-with-word-cloud-and-remove-empty-border

# Initialize wordcloud object
mgm_macau_review_wc = WordCloud(width=1500, height=1000,background_color='white', max_words=100,
        # update stopwords to include common words 
                                stopwords=STOPWORDS

#        stopwords = safe_get_stop_words('english').append 
#        STOPWORDS.update(['hotel','mgm','macau','macau','room','staff','hotels','friend','free','us'
#                                     ,'much','visit' , 'many' , 'check''even'
#                                     ,'night','casino','service','stayed','got'
#                                     ,'Venetian','trip','made','will', 'went','always','say','came','need'
#                                     ,'day','time','u','one','first'])
#               
)


# Generate and plot wordcloud
plt.imshow(mgm_macau_review_wc.generate(combined_text))
plt.figure(figsize=[20,10])
#plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.imshow(mgm_macau_review_wc)




#mgm_macau_reviews[mgm_macau_reviews['sentiment']==sentiment2]

sentiment2 = 'Negative'

# Combine all reviews for the desired sentiment
combined_text2 = " ".join([review for review in mgm_macau_reviews['review'][mgm_macau_reviews['sentiment']==sentiment2]])


#stopwords = nltk.corpus.stopwords.words('english')
STOPWORDSS = stopwords.words('english')
newStopWords2 = ['good','like','nice'',fact','could','really','would','checked','hotel','mgm','macau','macau','room','staff','hotels','friend','free','us','check'
                                     ,'still','aksed','ok','much','visit' ,'stay','said','ready','grand', 'many' , 'check','even','way','bit','arrived'
                                     ,'night','casino','service','stayed','got','around','provide','really','helpful','German'
                                     ,'every','even','make','check','took','provided','wynn','staying','booked'
                                     ,'Venetian','trip','made','will', 'went','always','say','came','need'
                                     ,'year','think','asked','excellent','told','today','quite','day','time','u','one','first','great','go','friendly','well','gave','get'
                                     "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"                                     
                                     ]
STOPWORDSS.extend(newStopWords2)



#width=800, height=400
#https://stackoverflow.com/questions/28786534/increase-resolution-with-word-cloud-and-remove-empty-border

# Initialize wordcloud object
mgm_macau_negreview_wc = WordCloud(width=1500, height=1000,background_color='white', max_words=100,
        # update stopwords to include common words 
                                   stopwords=STOPWORDSS
  
)


# Generate and plot wordcloud
plt.imshow(mgm_macau_negreview_wc.generate(combined_text2))
plt.figure(figsize=[20,10])
#plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.imshow(mgm_macau_negreview_wc)






sentiment3 = 'Natural'

# Combine all reviews for the desired sentiment
combined_text3 = " ".join([review for review in mgm_macau_reviews['review'][mgm_macau_reviews['sentiment']==sentiment3]])


#stopwords = nltk.corpus.stopwords.words('english')
STOPWORDSSS = stopwords.words('english')
newStopWords3 = ['good','like','nice','could','really','would','checked','hotel','mgm','macau','macau','room','staff','hotels','friend','free','us','check'
                                     ,'still','aksed','ok','much','visit' ,'stay','said','ready','grand', 'many' , 'check','even','way','bit','arrived'
                                     ,'night','casino','service','stayed','got','around','provide','really'
                                     ,'every','even','make','check','took','provided','wynn','staying','booked'
                                     ,'Venetian','trip','made','will', 'went','always','say','came','need'
                                     ,'year','think','asked','excellent','told','today','quite','day','time','u','one','first','great','go','friendly','well','gave','get'
                                     "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"                                     
                                     ]
STOPWORDSSS.extend(newStopWords3)



#width=800, height=400
#https://stackoverflow.com/questions/28786534/increase-resolution-with-word-cloud-and-remove-empty-border

# Initialize wordcloud object
mgm_macau_natreview_wc = WordCloud(width=1500, height=1000,background_color='white', max_words=100,
        # update stopwords to include common words 
                                   stopwords=STOPWORDSSS
  
)


# Generate and plot wordcloud
plt.imshow(mgm_macau_natreview_wc.generate(combined_text3))
plt.figure(figsize=[20,10])
#plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.imshow(mgm_macau_natreview_wc)






import numpy as np

##################### Dates manipulation
#https://www.geeksforgeeks.org/python-conditional-string-append/


#mgm_macau_reviews[mgm_macau_reviews['reviewdate'].str.slice(1,2)=='-']
#mgm_macau_reviews['reviewdate'].str.slice(1,2)=='-'
#mgm_macau_reviews.loc[(mgm_macau_reviews["reviewdate"].str.slice(1,2))=='-']
                 
mgm_macau_reviews['flag']=mgm_macau_reviews['reviewdate'].str.slice(1,2)     
#mgm_macau_reviews['reviewdate'].str.slice(1,2).isin(['-'])
#mgm_macau_reviews['reviewdate'].str.slice(1,2)
#mgm_macau_reviews['reviewdate'].str.slice(1,2).str.contains('-')




#https://datascience.stackexchange.com/questions/77816/valueerror-the-truth-value-of-a-series-is-ambiguous-after-applying-if-else-co
#df['col'] = 'str' + df['col'].astype(str)

#def append_str(item, add_str):
#     if  item.str.slice(1,2)=='-' :
   # if  lambda row: row.apply(item).astype(str).str.slice(1,2)=='-' :
    #if  np.where(item.str.slice(1,2).str.contains('-')):
        
    #if np.where(mgm_macau_reviews['reviewdate'].str.slice(1,2).astype(str)=='-'):
    #if np.where(mgm_macau_reviews['flag']=='-')
        #return add_str+item
         #return add_str.join(item)
#add_str='0'  

#mgm_macau_reviews['reviewdatev1']=lambda row: row.apply(append_str(mgm_macau_reviews['reviewdate'],add_str))
#append_str(mgm_macau_reviews['reviewdate'],add_str)
#mgm_macau_reviews['reviewdatev1'] =  map( append_str(mgm_macau_reviews['reviewdate'],add_str),mgm_macau_reviews)
#mgm_macau_reviews['reviewdatev1'] = append_str(mgm_macau_reviews['reviewdate'], add_str)



#mgm_macau_reviews['reviewdatev1']=mgm_macau_reviews['reviewdate']
#https://stackoverflow.com/questions/43830102/conditionally-append-string-to-rows-in-a-pandas-dataframe-based-on-presence-of-v

mgm_macau_reviews['reviewdatev1'] =mgm_macau_reviews.apply(lambda x: '0'+x.reviewdate if x.flag=='-' else x.reviewdate, axis=1)

       




import datetime

#.mgm_macau_reviews.year

mgm_macau_reviews['Dateofstay2'] = mgm_macau_reviews['Dateofstay'].str.lstrip()

mgm_macau_reviews['reviewdatev2']=pd.to_datetime(mgm_macau_reviews['reviewdatev1'],format='%y-%b') 

mgm_macau_reviews['Dateofstay2'] =pd.to_datetime(mgm_macau_reviews['Dateofstay2'],format='%B %Y') 



#datetime.datetime.strptime()

#df = ['19-Jan-19', '4-Jan-19']
#df['date_given'].dt.year




#https://www.shanelynn.ie/bar-plots-in-python-using-pandas-dataframes/




mgm_macau_reviews[['sentiment','review']].groupby(by=(['sentiment'])).count().plot(kind="bar")
#mgm_macau_reviews[['reviewdatev1','sentiment','review']].groupby(by=(['reviewdatev1','sentiment'])).count().plot(kind="bar")




mgm_macau_reviews['count']=1
mgm_macau_reviews_timeseries=mgm_macau_reviews[['reviewdatev2','review']].groupby(by=(['reviewdatev2'])).count().reset_index()

top_dates = mgm_macau_reviews_timeseries.sort_values(by=['review'],ascending=False).head(5)
vals = []
for tgl, tot in zip(top_dates["reviewdatev2"], top_dates["review"]):
    tgl = tgl.strftime("%b %y")
    #https://stackoverflow.com/questions/4288973/whats-the-difference-between-s-and-d-in-python-string-formatting
    val = "%d (%s)" % (tot, tgl)
    #print(val)
    vals.append(val)
top_dates['tgl'] = vals



#https://medium.com/nerd-for-tech/how-to-plot-timeseries-data-in-python-and-plotly-1382d205cc2

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

#https://stackoverflow.com/questions/61076090/plotly-figure-window-doesnt-appear-using-spyder
#https://plotly.com/python/renderers/
#conda install -c plotly plotly-orca==1.2.1
pio.orca.config.executable = 'C:/ProgramData/Anaconda3/pkgs/plotly-orca-1.2.1-1/orca_app/orca.exe'
#pio.renderers.default='browser'
pio.renderers.default = "svg"
#pio.renderers.default = "png"

fig = go.Figure(data=go.Scatter(x=mgm_macau_reviews_timeseries['reviewdatev2'].astype(dtype=str), 
                                y=mgm_macau_reviews_timeseries['review'],
                                marker_color='black', text="counts"))
fig.update_layout({"title": 'Reviews from trip.com from 2008 to 2021',
                   "xaxis": {"title":"Date"},
                   "yaxis": {"title":"Total Reviews"},
                   "showlegend": False})
fig.add_traces(go.Scatter(x=top_dates['reviewdatev2'], y=top_dates['review'],
                          textposition='top right',
                          textfont=dict(color='#233a77'),
                          mode='markers+text',
                          marker=dict(color='red', size=6),
                          text = top_dates["tgl"]
                          )
                          )
fig.show()





import seaborn as sns


num_words = []
for x,word in enumerate(mgm_macau_reviews['review']):
    num_words1 = len(word.split())
    #print(num_words1)
    num_words.append(num_words1)
mgm_macau_reviews['cnt']=num_words



#from collections import Counter
wordcount = mgm_macau_reviews['cnt']
sns.distplot(wordcount, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
sns.distplot(mgm_macau_reviews['sentiment'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

#https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
sentiment = ['Positive', 'Negative']
# Iterate through the five airlines
for sentiment in sentiment:
    # Subset to the airline
    subset = mgm_macau_reviews[mgm_macau_reviews['sentiment'] == sentiment]
    
    # Draw the density plot
    sns.distplot(subset['cnt'], hist = False, kde = True,
                 #kde_kws = {'linewidth': 3},
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 label = sentiment)

# Plot formatting
plt.legend(prop={'size':9}, title = 'sentiment')    
plt.title('Density Plot with Multiple Sentiment')
plt.xlabel('Review Word Count')
plt.ylabel('Density')






"""
# Preprocessing v1
#imbalance
https://datauab.github.io/sentiment_predictions/
"""




"""
# Preprocessing v1
#Here, if we consider only unigrams, then the single word cannot convey the details properly. If we have a word like ‘Machine learning developer’, then the word extracted should be ‘Machine learning’ or ‘Machine learning developer’. The words simply ‘Machine’, ‘learning’ or ‘developer’ will not give the expected result.
"""


import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




if 'STOPWORDS_sentiment' in locals():
    del STOPWORDS_sentiment
    
STOPWORDS_sentiment = (stopwords.words('english'))
newStopWords = ['cotai','including','experienced','experience','macao','covid','pandemic','hotel','mgm','las','vegas','hong kong','hong','kong','MGM','Grand','Macau','macau','room','staff','hotels','friend','free','us','check'
                                     ,'much','visit' , 'many' , 'check','even','way','bit','arrived'
                                     ,'night','casino','service','stayed','got','around','provide','really'
                                     ,'every','even','make','check','took','provided','wynn','staying','booked'
                                     ,'Venetian','trip','made','will', 'went','always','say','came','need'
                                     ,'day','time','u','one','first'
                                     "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
STOPWORDS_sentiment.extend(newStopWords)


from nltk.corpus import names
import random
# Construct a list of classified names, using the names corpus.
namelist = [(name) for name in names.words('male.txt')] + [(name) for name in names.words('female.txt')]
random.seed(123456)
random.shuffle(namelist)
    


#geeksforgeeks.org/tf-idf-for-bigrams-trigrams/    
#TF-IDF in NLP stands for Term Frequency – Inverse document frequency. It is a very popular topic in Natural Language Processing which generally deals with human languages. During any text processing, cleaning the text (preprocessing) is vital. Further, the cleaned data needs to be converted into a numerical format where each 
#word is represented by a matrix (word vectors). This is also known as word embedding
#Term Frequency (TF) = (Frequency of a term in the document)/(Total number of terms in documents)
#Inverse Document Frequency(IDF) = log( (total number of documents)/(number of documents with term t))
#TF.IDF = (TF).(IDF)





"""
# Preprocessing
#bigrams-trigrams
#https://www.geeksforgeeks.org/tf-idf-for-bigrams-trigrams/
"""

def remove_string_special_characters(s):
      
    # removes special characters with ' '
    stripped = re.sub('[^a-zA-z\s]', '', s)
    stripped = re.sub('_', '', stripped)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)
      
    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
            return stripped.lower()

# Stopword removal 
#stop_words = set(stopwords.words('english'))
#your_list = ['x', 'y', 'n', 'z']
for i, line in enumerate(mgm_macau_reviews.review):
    mgm_macau_reviews.review[i] = ' '.join([x for 
        x in nltk.word_tokenize(line) if
        (x.lower() not in STOPWORDS_sentiment) and ( x.lower() not in namelist )])
    #print(i, line)

         

## Getting trigrams 
#vectorizer = CountVectorizer(ngram_range = (3,3))
#X1 = vectorizer.fit_transform(mgm_macau_reviews.review) 
#features = (vectorizer.get_feature_names())
##print("\n\nFeatures : \n", features)
##print("\n\nX1 : \n", X1.toarray())
  
## Applying TFIDF
#vectorizer = TfidfVectorizer(ngram_range = (3,3))
#X2 = vectorizer.fit_transform(mgm_macau_reviews.review)
#scores = (X2.toarray())
##print("\n\nScores : \n", scores)
  
## Getting top ranking features
#sums = X2.sum(axis = 0)
#data1 = []
#for col, term in enumerate(features):
#    data1.append( (term, sums[0,col] ))
#ranking = pd.DataFrame(data1, columns = ['term','rank'])
#words = (ranking.sort_values('rank', ascending = False))
#print ("\n\nWords head : \n", words.head(50))



# Getting bigrams 
vectorizer = CountVectorizer(ngram_range =(2, 2))
X1 = vectorizer.fit_transform(mgm_macau_reviews.review) 
features = (vectorizer.get_feature_names())
print("\n\nX1 : \n", X1.toarray())

# Applying TFIDF
# You can still get n-grams here
vectorizer = TfidfVectorizer(ngram_range = (2, 2))
X2 = vectorizer.fit_transform(mgm_macau_reviews.review)
scores = (X2.toarray())
print("\n\nScores : \n", scores)
  
# Getting top ranking features
sums = X2.sum(axis = 0)
print(sums)
data1 = []
for col, term in enumerate(features):
    data1.append( (term, sums[0, col] ))
ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
words = (ranking.sort_values('rank', ascending = False))
print ("\n\nWords : \n", words.head(50))



#https://kanoki.org/2019/11/06/python-detect-and-translate-language/




def get_word_features(wordlist):
  wordlist = nltk.FreqDist(wordlist)
  word_features = [w for (w, c) in wordlist.most_common(200)] 
  return word_features  



combined_text_biggram = " ".join([review for review in features])   
plt.imshow(mgm_macau_negreview_wc.generate(combined_text_biggram))









# BeautifulSoup to easily remove HTML tags
from bs4 import BeautifulSoup 

# RegEx for removing non-letter characters
import re

from nltk.stem.porter import *
#wordcoreextractor
stemmer = PorterStemmer()




debug=0
def log(text):
    if debug>0:
        print(text)

def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    
    # TODO: Remove HTML tags and non-letters,
    #       convert to lowercase, tokenize,
    #       remove stopwords and stem
    log('Raw input :')
    log(review)
    
    # remove HTML tags
    review=BeautifulSoup(review, "html5lib").get_text()
    #log('\nHTML tags removed')
    #log(review)
    
    # remove punctuation and numeric
    #review = re.sub(r"[^a-zA-Z0-9]", " ", review) 
    review = re.sub(r"[^a-zA-Z]", " ", review) 
    log('\n Punctuation removed')
    log(review)
    
    # lowercase    
    review = review.lower()
    
    # tokenize
    review = nltk.word_tokenize(review)
    log('\n Tokenized')
    log(review)
    
    # remove stop words
    #review = [w for w in review if w not in stopwords.words("english")]
    review = [w for w in review if w not in STOPWORDS_sentiment and w not in ([n.lower() for n in namelist])]
    log('\n Stop words removed')
    log(review)
    
    # stemming
    #review = [PorterStemmer().stem(w) for w in review if w not in STOPWORDS_sentiment and w not in ([n.lower() for n in namelist])]
    #log('\n Stemmed')
    #log(review)
    
    log('\n\n')
    #words=[]
    
    # Return final list of words
    return review




import pickle
import os




cache_dir = os.path.join("Z:/MGM/My work/Python/bin/","cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train,data_test,
                    cache_dir=cache_dir, cache_file="preprocessed_mgm_reviewdata.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        words_train = list(map(review_to_words, data_train))
        words_test = list(map(review_to_words, data_test))
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        #words_train = (cache_data['words_train'])
        words_train, words_test = (cache_data['words_train'],
                cache_data['words_test'])
    
    return words_train, words_test


# Preprocess data





words_train, words_test= preprocess_data(mgm_macau_reviews.review,mgm_cotai_reviews.review)
#words_train, words_test= preprocess_data(mgm_macau_reviews["review"],mgm_cotai_reviews["review"])


os.remove (os.path.join(cache_dir,'preprocessed_mgm_reviewdata.pkl'))

# Take a look at a sample
print("\n--- Raw review ---")
print(mgm_macau_reviews['review'][0])
print("\n--- Tokenized ---")    
print(nltk.word_tokenize(mgm_macau_reviews['review'][0]))
print("\n--- Preprocessed words ---")
print(words_train[0])
print("\n--- Label ---")
print(mgm_macau_reviews["sentiment"][1])


print("\n--- Raw review ---")
print(mgm_cotai_reviews['review'][0])
print("\n--- Preprocessed words ---")
print(words_test[0])
print("\n--- Label ---")
print(mgm_cotai_reviews["sentiment"][1])



#remove names
#from nltk.tag.stanford import StanfordNERTagger
#st = StanfordNERTagger("Z:/MGM/My work/Python/NLP/classifiers/english.all.3class.distsim.crf.ser.gz"
#                       , "Z:/MGM/My work/Python/NLP/jar/stanford-ner.jar")
#text = """The front desk staff were super nice. You experienced great service right after you stepped into the lobby. Jason was the one who served us, super friendly. We got our room upgraded for free, it was so nice and warm-hearted"""
#for sent in nltk.sent_tokenize(text):
#    tokens = nltk.tokenize.word_tokenize(text)
#    tags = st.tag(tokens)
#    for tag in tags:
#        if tag[1]=='PERSON': print(tag)



"""
# Post-processing
# Feature Extraction
#Extracting Bag-of-Words 
#Compute Bag-of-Words features¶
https://dinghe.github.io/sentiment_analysis.html
"""


"""
resampling
"""
from imblearn.combine import SMOTETomek
smt = SMOTETomek(sampling_strategy='auto')
smt = SMOTETomek(random_state=42)





from sklearn.feature_extraction.text import CountVectorizer
import joblib
#import sklearn.external.joblib as extjoblib

# joblib is an enhanced version of pickle that is more efficient for storing NumPy arrays


"""
original without resampling

def extract_BoW_features(words_train, words_test, vocabulary_size=10000,
                         cache_dir=cache_dir, cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # TODO: Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        #vectorizer = CountVectorizer(preprocessor=lambda x: x,tokenizer=lambda x: x, lowercase=True,max_features=5000)
        vectorizer = CountVectorizer(preprocessor=lambda x: enumerate(x),tokenizer=lambda x: enumerate(x), lowercase=True,max_features=300)
        features_train = vectorizer.fit_transform(words_train).toarray()
        # TODO: Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test = vectorizer.fit_transform(words_test).toarray()
              
        # NOTE: Remember to convert the features using .toarray() for a compact representation
        
        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test ,
                             vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                cache_data['features_test'], cache_data['vocabulary'])
    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary


# Extract Bag of Words features for both training and test datasets

#y=mgm_macau_reviews["sentiment"]
#features_train, y = smt.fit_resample(features_train,y )


features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test)

"""



"""
with resampling
"""

y=mgm_macau_reviews["sentiment"]
y2=mgm_cotai_reviews["sentiment"]
def extract_BoW_features(words_train, words_test, vocabulary_size=10000,
                         cache_dir=cache_dir, cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # TODO: Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(preprocessor=lambda x: x,tokenizer=lambda x: x, lowercase=True,max_features=300)
        #vectorizer = CountVectorizer(preprocessor=lambda x: enumerate(x),tokenizer=lambda x: enumerate(x), lowercase=True,max_features=1000)
        features_train = vectorizer.fit_transform(words_train).toarray()
        # TODO: Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test = vectorizer.fit_transform(words_test).toarray()
         
        features_train, y_smt = smt.fit_resample(features_train,y)
        features_test, y2_smt = smt.fit_resample(features_test,y2)
        
        # NOTE: Remember to convert the features using .toarray() for a compact representation
        
        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test ,
                             vocabulary=vocabulary,y_smt=y_smt,y2_smt=y2_smt)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                cache_data['features_test'], cache_data['vocabulary']) , y_smt,y2_smt
    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary, y_smt,y2_smt


# Extract Bag of Words features for both training and test datasets

#y=mgm_macau_reviews["sentiment"]
#features_train, y = smt.fit_resample(features_train,y )


features_train, features_test, vocabulary, y_smt,y2_smt = extract_BoW_features(words_train, words_test)


os.remove (os.path.join(cache_dir,'bow_features.pkl'))





# Inspect the vocabulary that was computed
print("Vocabulary: {} words".format(len(vocabulary)))

#import random
print("Sample words: {}".format(random.sample(list(vocabulary.keys()), 8)))

# Print sample
print("\n--- Preprocessed words ---")
print(words_train[0])
print("\n--- Bag-of-Words features ---")
print(features_train[0])
print("\n--- Label ---")
print(mgm_cotai_reviews["sentiment"][0])


# Putting all together
print("\n--- Raw review ---")
print(mgm_macau_reviews['review'][0])
print("\n--- Preprocessed words ---")
print(words_train[0])
print("\n--- Bag-of-Words features ---")
print(features_train[0])
print("\n--- Label ---")
print(mgm_macau_reviews["sentiment"][1])



#y=['desk super nice . great stepped lobby . Jason served , super friendly . upgraded , nice warm-hearted']
#vectorizer = CountVectorizer(preprocessor=lambda x: enumerate(x),tokenizer=lambda x: enumerate(x), lowercase=True,max_features=1000)
#vectorizer = CountVectorizer(preprocessor=lambda x: x,tokenizer=lambda x: x, lowercase=True,max_features=2000)

#xxx = vectorizer.fit_transform(y).toarray()



#corpus = [
#   'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(corpus)
#print(vectorizer.get_feature_names())
#xxxx=(X.toarray())



[index for index in features_train[1] if index != 0]


# Plot the BoW feature vector for a training document
plt.plot(features_train[67,:])
plt.xlabel('Word')
plt.ylabel('Count')
plt.show()

# Find number of occurrences for each word in the training set
word_freq = features_train.sum(axis=0)

# Sort it in descending order
sorted_word_freq = np.sort(word_freq)[::-1]

# Plot 
plt.plot(sorted_word_freq)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel('Rank')
plt.ylabel('Number of occurrences')
plt.show()









#import seaborn as sns
target_count = mgm_macau_reviews['sentiment'].value_counts()
print("Positive:",  target_count[0])
print("Negative:",  target_count[1])
def plot_distributionCount(x, data):
    #plt.subplots(figsize=(6, 4))
    sns.countplot(x=x, data=data, order=data[x].value_counts().index, palette='icefire_r')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.show()  
plot_distributionCount('sentiment',mgm_macau_reviews)
plot_distributionCount('sentiment',pd.DataFrame(y_smt))






my_dpi =100
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10), dpi=my_dpi)
print(fig)
#print(axes)

# title for entire figure
fig.suptitle('Before and After Resampling', fontsize=20)


# edit subplots
ax[0].set_title('Before', fontsize=14)
ax[1].set_title('After', fontsize=14)


sns.countplot(mgm_macau_reviews['sentiment'], ax=ax[0])
sns.countplot(y_smt, ax=ax[1])







"""
#Normalize feature vectors
https://dinghe.github.io/sentiment_analysis.html
"""

import sklearn.preprocessing as pr

# TODO: Normalize BoW features in training and test set
features_train=pr.normalize(features_train, norm='l2',copy=False)
features_test=pr.normalize(features_test, norm='l2',copy=False)

#[index for index in features_train[5] if index != 0]



"""
#Pad sequences
In order to feed this data into your RNN, all input documents must have the same length. Let's limit the maximum review length to max_words by truncating longer reviews and padding shorter reviews with a null value (0). You can accomplish this easily using the pad_sequences() function in Keras. For now, set max_words to 500.


from keras.preprocessing import sequence

# Set the maximum number of words per document (for both training and testing)
max_words = 500

# TODO: Pad sequences in X_train and X_test
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

https://dinghe.github.io/sentiment_analysis.html
"""







"""
# Building Machine Learning Models
#ML Classification using BoW features
https://dinghe.github.io/sentiment_analysis.html

"""


"""
1.Gaussian Naïve Bayes
"""

from sklearn.naive_bayes import GaussianNB

# TODO: Train a Guassian Naive Bayes classifier
clf1 = GaussianNB().fit(features_train,mgm_macau_reviews.sentiment)

# Calculate the mean accuracy score on training and test sets
print("[{}] Accuracy: train = {}, test = {}".format(
        clf1.__class__.__name__,
        clf1.score(features_train,mgm_macau_reviews.sentiment)
        ,clf1.score(features_test,mgm_cotai_reviews.sentiment)
        ))


"""
with resampling
"""
clf1 = GaussianNB().fit(features_train,y_smt)
# Calculate the mean accuracy score on training and test sets
print("[{}] Accuracy: train = {}, test = {}".format(
        clf1.__class__.__name__,
        clf1.score(features_train,y_smt)
        ,clf1.score(features_test,y2_smt)
        ))


# TODO: Write a sample review and set its true sentiment
my_review = "This hotel is absolutely depressing"
true_sentiment = 'Negative'

# TODO: Apply the same preprocessing and vectorizing steps as you did for your training data
my_words = review_to_words(my_review)
vectorizer = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)
my_features = vectorizer.transform(my_words).toarray()
my_features = pr.normalize(my_features)
# TODO: Then call your classifier to label it
yhat = clf1.predict(my_features)
print(yhat)


# TODO: Write a sample review and set its true sentiment
my_review = "Greet friendly staffs 4-hours flight heavy traffic . desk staffs friendly smiling . ordered cake advance celebrate boyfriend ’ birthday , arranged prepared addition souvenirs . recommended families travel , great people excellent hospitality feel welcomed . choose future ."
true_sentiment = 'Positive' 

# TODO: Apply the same preprocessing and vectorizing steps as you did for your training data
my_words = review_to_words(my_review)
#vectorizer = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: enumerate(x), tokenizer=lambda x: enumerate(x), lowercase=True)
vectorizer = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)
my_features = vectorizer.transform(my_words).toarray()
my_features = pr.normalize(my_features)
# TODO: Then call your classifier to label it
yhat = clf1.predict(my_features)
print(yhat[-1])



"""
2.Gradient-Boosted Decision Tree classifier
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

n_estimators = range(20, 81, 10)

def classify_gboost(X_train, X_test, y_train, y_test):        
    # Initialize classifier
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=1, random_state=0)

    # TODO: Classify the data using GradientBoostingClassifier
    gsearch = GridSearchCV(estimator = clf, param_grid = {'n_estimators': n_estimators}, verbose=50, n_jobs=-1)
    gsearch.fit(X_train, y_train)
    # TODO(optional): Perform hyperparameter tuning / model selection
    best_clf = gsearch.best_estimator_
    # TODO: Print final training & test accuracy
    print(gsearch.best_params_)
    # Return best classifier model
    print("[{}] Accuracy: train = {}, test = {}".format(
    best_clf.__class__.__name__,
    best_clf.score(X_train, y_train),
    best_clf.score(X_test, y_test)))
    
    return best_clf



clf2 = classify_gboost(features_train, features_test, mgm_macau_reviews["sentiment"], mgm_cotai_reviews["sentiment"])

"""
with resampling
"""
clf2 = classify_gboost(features_train, features_test, y_smt, y2_smt)



# Calculate the mean accuracy score on training and test sets
print("[{}] Accuracy: train = {}, test = {}".format(
        clf2.__class__.__name__,
        clf2.score(features_train,y_smt)
        ,clf2.score(features_test,y2_smt)
        ))


# TODO: Write a sample review and set its true sentiment
my_review = "This hotel is noise,hate, bad, slow, horrible,rude, pool is unavailable"
#my_review ="Requested for higher floor and end-up with 7 floors room~ Ok! Fine! It’s summer time and the pool is unavailable ~ Ok! Fine! Let’s take a look at the bathtub!! Unbelievable!! Look at those dirt, dust and hair!! No more MGM!! Never ever!!"

true_sentiment = 'Negative'

# TODO: Apply the same preprocessing and vectorizing steps as you did for your training data
my_words = review_to_words(my_review)
vectorizer = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: enumerate(x), tokenizer=lambda x: enumerate(x), lowercase=True)
my_features = vectorizer.transform(my_words).toarray()

#my_features = pr.normalize(my_features)
# TODO: Then call your classifier to label it
yhat = clf2.predict(my_features)
print(yhat)



# TODO: Write a sample review and set its true sentiment
#my_review = "chose celebrate start summer holiday . availed Local package dining credit $ 800 practical family spending . , staffs informative answering inquiries ideas places . rooms spacious nice view overlooking Taipa island amazing amenities good Adults Kids alike . worth , surely recommend friends ."
my_review ="Our room got upgraded with complimentary wine and fruits, spacious room and amenity controlled by smart technology, it’s a new experience. The facilities, like swimming pool and spa, are new and staff are kind and helpful."
#my_review = "Booked a staycation package for my birthday celebration this is. Such a great deal amongst the other 5 star hotels. Had my birthday lunch at Grill 58. Amazing seafood on ice and unforgettable steak 😍. The bread and butter is really unique. Never had such delicious bread and butter before. Had a surprise bday cake to end our perfect lunch. Thank u Joseph for the wonderful arrangements in the room and all the birthday surprises. Joseph was really attentive to my requests and provided an exceptional service.Had such a great and relaxing time there. The breakfast buffet was really good too. They serve quality food. Even had eggs benedicts (my fav). Siew mai and congee is soooo good. Drop by the pool before check out. Pool attendants were really attentive and nice. They serve little…"
#my_review = "Our room got upgraded with complimentary wine and fruits, spacious room and amenity controlled by smart technology, it’s a new experience. The facilities, like swimming pool and spa, are new and staff are kind and helpful."
#my_review ="Some years ago we had a stay at MGM and were attracted by the giant aquarium. The colourful fishes and the coral reefs were so eye catching. We had a great stay there and got a little lion soft toy that we still keep.  This time we tried another MGM hotel at Cotai. Of course,Front desk attendant Daniel, good service and the decorations are spectacular and the atmosphere is so relaxing and comfortable.  What is the best is different cuisines are provided and they all offer good service and the staff are friendly too.  It’s really good to spend a relaxing holiday in this hotel. "
#my_review ="Staff was extremely friendly and provided great service, stayed for one night with a friend and ate some in room dining. Their menu has a great variety of options including breakfast sets and also great quality of food. Would definitely choose MGM Cotai again. In addition the views from the room was also magnificent!"
#my_review ="Stay here to celebrate my sister’s birthday, thank you for Ryan and Vita help me to upgrade my room and the wonderful arrangements for my sister. This is my third visit to this hotel, each time has wonderful and nice experience. I will come again, thank you to all the staff of MGM Cotai."



#my_review ="Service were excellent! Ms.man's service are great!! Strongly recommend to stay here for special events or special dates. Janice Wong's store are great, especially the black seaseme ice cream! Must try item!"


true_sentiment = 'Positive'



# TODO: Apply the same preprocessing and vectorizing steps as you did for your training data

my_words = review_to_words(my_review)
my_words=[my_words]
#vectorizer = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: enumerate(x), tokenizer=lambda x: enumerate(x), lowercase=True)
#vectorizer = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)
#vectorizer = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)


vectorizer = CountVectorizer(vocabulary=vocabulary,preprocessor=lambda x:x,tokenizer=lambda x:x, lowercase=True)

#vectorizer = CountVectorizer(vocabulary=vocabulary,preprocessor=lambda x: enumerate(x),tokenizer=lambda x: enumerate(x))


#vectorizer = CountVectorizer(vocabulary=vocabulary,preprocessor=lambda x: enumerate(x),tokenizer=lambda x: enumerate(x), lowercase=True,max_features=1000)
my_features = vectorizer.fit_transform(my_words).toarray()


#my_features = vectorizer.transform(my_words).toarray()
#my_features = vectorizer.fit_transform(words_train).toarray()

#my_features = pr.normalize(my_features)
# TODO: Then call your classifier to label it
yhat = clf2.predict(my_features)
print(yhat)




for n in range(len(my_features[0:])):
    print([index for index in my_features[0] if index != 0])


#corpus = [
#   'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]
#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(corpus)
#print(vectorizer.get_feature_names())
#xxxx=(X.toarray())










"""
#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# Save your model, so that you can quickly load it in future (and perhaps resume training)
model_file = "rnn_model.h5"  # HDF5 file
model.save(os.path.join(cache_dir, model_file))


with open('Z:/MGM/My work/Python/bin/cache/sentiment_analysis', 'wb') as f:
  pickle.dump(clf2, f)
  
pickle.dump(clf2, open("x.pickle.dat", "wb"))

print("Saved model to: pima.joblib.dat")
 



with open('Z:/MGM/My work/Python/bin/cache/sentiment_analysis', 'wb') as f:

pickle.dump(vocabulary, open("vocabulary.dat", "wb"))

# some time later...
 
#import torch
# load model from file
#loaded_model = pickle.load("Z:/MGM/My work/Python/bin/cache/sentiment_analysis/pima.pickle.dat")

loaded_model = pickle.load(open('Z:/MGM/My work/Python/bin/cache/sentiment_analysis/pima.pickle.dat', 'rb'))

print("Loaded model from: pima.joblib.dat")

vocabulary2 = pickle.load(open('Z:/MGM/My work/Python/bin/cache/sentiment_analysis/vocabulary.dat', 'rb'))
    

# Later you can load it using keras.models.load_model()
from keras.models import load_model
model = load_model(os.path.join(cache_dir, model_file))


#Test
print(loaded_model.predict(my_features))


original_text ="Our room got upgraded with complimentary wine and fruits, spacious room and amenity controlled by smart technology, it’s a new experience. The facilities, like swimming pool and spa, are new and staff are kind and helpful."

def summarize2(original_text):
    my_words=BeautifulSoup(original_text, "html5lib").get_text()

    my_words = re.sub(r"[^a-zA-Z]", " ", my_words) 

    my_words = my_words.lower()

    my_words = nltk.word_tokenize(my_words)

    my_words = [w for w in my_words if w not in STOPWORDS_sentiment and w not in ([n.lower() for n in namelist])]
    
    my_words = review_to_words(original_text)
    #here
    my_words2=my_words
    #my_words2=','.join(str(e) for e in my_words)
    my_words=[my_words]
    vectorizer = CountVectorizer(vocabulary=vocabulary2,preprocessor=lambda x:x,tokenizer=lambda x:x, lowercase=True)
    my_features = vectorizer.fit_transform(my_words).toarray()
    yhat = loaded_model.predict(my_features)
    #return format(my_words2)
    return my_words2

type(summarize2(original_text))


"""








"""
3.RNN
https://dinghe.github.io/sentiment_analysis.html#Step-5:-Switching-gears---RNNs
"""


from keras.datasets import imdb  # import the built-in imdb dataset in Keras

# Set the vocabulary size
vocabulary_size = 5000

# Load in training and test data (note the difference in convention compared to scikit-learn)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print("--- Review (with words) ---")
print([id2word.get(i, " ") for i in X_train[7]])
print("--- Label ---")
print(y_train[7])



from keras.preprocessing import sequence

# Set the maximum number of words per document (for both training and testing)
max_words = 500

# TODO: Pad sequences in X_train and X_test
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

#Muti-GPU training
import tensorflow as tf
#server = tf.train.server.create_local_server()
#sess = tf.Session(server.target)
from keras import backend as K
#K.set_session(sess)

# TODO: Design your model
model = Sequential()
model.add(Embedding(vocabulary_size, 128, input_length=max_words))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())





logisticRegression = LogisticRegression()
naiveBayes = MultinomialNB()
SVM = SVC()
randomForest = RandomForestClassifier(n_estimators=50)
neuralNetwork = MLPClassifier()

models = [logisticRegression, naiveBayes, SVM, randomForest, neuralNetwork]










"""
Desemination visulastion and dashing
"""


#https://www.datarevenue.com/en-blog/data-dashboarding-streamlit-vs-dash-vs-shiny-vs-voila
import dash
import dash_core_components as dcc
import dash_html_components as html


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig),
])

app.run_server()



C:\Users\119987\dash-sample-apps-main\dash-sample-apps-main\apps\dash-summarize
python app.py

#https://jsiwu94.github.io/Airlines-on-Twitter/
#Customer sentiment:
#To identify positive and negative opinions, emotions, and evaluations.
#Identify negative topics:
#Derive negative topics that people are likely to mention when talking with their experience with airlines.
#Derive actionable insights:
#Insights may later be used by airlines in planning and execution of customer service initiatives, media relationships etc.


#You and I would have understood that sentence in a fraction of a second. But machines simply cannot process text data in raw form. They need us to break down the text into a numerical format that’s easily readable by the machine (the idea behind Natural Language Processing!).

#https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python
#Now the question - why Naive Bayes?
#You chose to study Naive Bayes because of the way it is designed and developed. Text data has some practicle and sophisticated features which are best mapped to Naive Bayes provided you are not considering Neural Nets. Besides, it's easy to interpret and does not create the notion of a blackbox model.
#Naive Bayes suffers from a certain disadvantage as well:
#The main limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that you get a set of predictors which are entirely independent.
#Why is sentiment analysis so important?
#Sentiment analysis solves a number of genuine business problems:
#It helps to predict customer behavior for a particular product.
#It can help to test the adaptability of a product.
#Automates the task of customer preference reports.
#It can easily automate the process of determining how well did a movie run by analyzing the sentiments behind the movie's reviews from a number of platforms.
#And many more!

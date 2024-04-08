from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV

"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""
# numpy and pandas are also permitted
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# basic python imports are permitted
import sys
import csv
import re
import random

data = pd.read_csv("clean_dataset.csv")

# TODO: CLEAN DATA 
# SECTION CLEAN DATA
#
#
#
#

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float("nan").
    """

    if isinstance(s, str):
        s = s.replace(",", "")
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """

    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1


def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

#  SECTION
#  LINEAR REGRESSION FUNCTIONS
#
#
#
#
#

def softmax(z):
    """
    Compute the softmax of vector z, or row-wise for a matrix z.
    For numerical stability, subtract the maximum logit value from each
    row prior to exponentiation (see above).

    Parameters:
        `z` - a numpy array of shape (K,) or (N, K)

    Returns: a numpy array with the same shape as `z`, with the softmax
        activation applied to each row of `z`
    """
    # print("softmax", z.shape)
    m = np.max(z, axis=1, keepdims=True)
    exp_elements = np.exp(z - m)
    sum_exp_elements = np.sum(exp_elements, axis=1, keepdims=True)
    y_k = exp_elements / sum_exp_elements

    return y_k

def pred(X, w):
    # print(X.shape, w.shape)
    z = np.matmul(X, w)  
    return softmax(z) 

def make_onehot(indicies):
    I = np.eye(4)
    return I[indicies]

def grad(w, X, t):
    """
    Return the gradient of the cost function at `w`. The cost function
    is the average cross-entropy loss across the data set `X` and the
    target vector `t`.

    Parameters:
        `w` - a current "guess" of what our weights should be,
                   a numpy array of shape (D+1)
        `X` - matrix of shape (N,D+1) of input features
        `t` - target y values of shape (N)

    Returns: gradient vector of shape (D+1)
    """

    y = pred(X, w)
    t_copy = t.copy()
    t_copy = make_onehot(t_copy)

    term1 = (y - t_copy) # (50, 4) (50, ) convert t to one hot vector should fix TIP: remind group members not to do global changes to da data

    n = len(t)

    return np.matmul(np.transpose(X), term1)/n

def solve_via_sgd(X_train, t_train, alpha=0.01, n_epochs=0, batch_size=50):
    """
    Given `alpha` - the learning rate
          `n_epochs` - the number of **epochs** of gradient descent to run
          `batch_size` - the size of ecach mini batch
          `X_train` - the data matrix to use for training
          `t_train` - the target vector to use for training
          `X_valid` - the data matrix to use for validation
          `t_valid` - the target vector to use for validation
          `w_init` - the initial `w` vector (if `None`, use a vector of all zeros)
          `plot` - whether to track statistics and plot the training curve

    Solves for logistic regression weights via stochastic gradient descent,
    using the provided batch_size.

    Return weights after `niter` iterations.
    """
    # as before, initialize all the weights to zeros
    w = np.zeros((X_train.shape[1], 4))

    # we will use these indices to help shuffle X_train
    N = X_train.shape[0] # number of training data points
    indices = list(range(N))

    for e in range(n_epochs):
        # Each epoch will iterate over the training data set exactly once.
        # At the beginning of each epoch, we need to shuffle the order of
        # data points in X_train. Since we do not want to modify the input
        # argument `X_train`, we will instead randomly shuffle the `indices`,
        # and we will use `indices` to iterate over the training data
        random.shuffle(indices)

        for i in range(0, N, batch_size):
            if (i + batch_size) >= N:
                continue

            # TODO: complete the below code to compute the gradient
            # only across the minibatch:
            indices_in_batch = indices[i: i+batch_size]
            X_minibatch = np.take(X_train, indices_in_batch, 0) # TODO: subset of "X_train" containing only the
                                                                #       rows in indices_in_batch
            #print(X_minibatch.shape)
            t_minibatch = np.take(t_train, indices_in_batch, 0) # TODO: corresponding targets to X_minibatch

            dw = grad(w, X_minibatch, t_minibatch) # TODO: gradient of the avg loss over the minibatch
            w = w - alpha * dw

    return w

def accuracy(w, X, t):
    y_pred = pred(X, w)  
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = np.mean(y_pred_classes == t)
    return accuracy

# Process MLE of Q10

vocab = ["racist", "doctor", "whitewashed", "playing", "realization", "specifically", "elderado", "symphony", "harvey", "second", "underground", "parent", "eagle", "keys", "brazilâ", "endless", "parisian", "day", "definitely", "son", "emptiness", "ohâ", "usa", "â", "leader", "forget", "community", "movies", "fashionable", "slice", "means", "beyond", "used", "particular", "charles", "money", "wrenching", "tomato", "builds", "tallest", "anymore", "trade", "wine", "real", "hearts", "clan", "train", "hollow", "enjoying", "heist", "apple", "painting", "seen", "unlike", "build", "mian", "hit", "janeiro", "romanceâ", "pizzas", "replies", "miraculous", "become", "theres", "eurocentric", "understand", "spectacular", "sleep", "goes", "okay", "olympia", "si", "asked", "inspires", "garbage", "world", "insideâ", "slavery", "share", "aha", "airlines", "newww", "kind", "wealth", "keep", "worlds", "social", "toilettes", "tells", "burj", "burj khalifa", "baguette", "festival", "rigorous", "tyson", "early", "love", "vests", "opportunity", "white", "complicated", "cars", "start", "home", "mood", "aldo", "nowhereâ", "heart", "games", "strikes", "enjoy", "thanks", "mind", "authentic", "wants", "nan", "psg", "bedbugs", "monsieur", "bigger", "crazy", "jobs", "bought", "pissed", "brasilâ", "elfâ", "five", "fk", "forests", "glitters", "instantly", "ó", "-", "glamour", "um", "though", "underbelly", "main", "outdated", "differences", "dude", "bayside", "wearing", "devouring", "mafia", "lifestyle", "wins", "whatâ", "mexicans", "streets", "labour", "romantic", "else", "ðÿžµwhen", "help", "stars", "light", "fancy", "possible", "chaois", "together", "wallet", "york", "daring", "beauty", "men", "cats", "birth", "talk", "habibi", "trendy", "song", "gather", "lively", "different", "nothin", "stock", "amazing", "kanye", "baguettes", "esque", "patisserie", "paradise", "enthusiastic", "wolf", "designating", "girl", "died", "peaceful", "might", "lot", "high", "needs", "super", "presenting", "charo", "best", "knowledgeable", "plate", "whole", "safe", "eleganceâ", "feelings", "magic", "parrots", "brothas", "kent", "low", "wanted", "gasoline", "bonjour", "breads", "crunches", "nightlife", "hon", "long", "making", "gustave", "jules", "americans", "É", "belfort", "necessite", "og", "rich", "discover", "air", "gets", "washington", "emily", "patagonia", "fullest", "sounds", "everyone", "extravagant", "dont", "taxis", "million", "khalifa", "inhabitants", "suite", "death", "clothing", "disney", "tower", "opportunities", "god", "countries", 
         "bright", "place", "lives", "earth", "thing", "fly", "house", "hereâ", "act", "drop", "richness", "á", "capital", "teenage", "hosted", "newyork", "waiting", "member", "destination", "corrupts", "unity", "khaled", "shaffer", "sea", "french", "flash", "give", "flying", "bateman", "coreta", "wisdom", "fall", "jordan", "awayâ", "j", "memories", "walkin", "nothing", "weather", "francais", "nymar", "may", "birds", "settle", "break", "dream", "M", "price", "exists", "oliveira", "w", "extremely", "viva", "impossible", "joy", "succession", "vida", "bread", "dance", "case", "siuuuu", "containing", "insatiate", "staying", "music", "chocolate", "moment", "classical", "take", "historic", "museum", "vein", "travel", "change", "shy", "book", "croissant", "david", "hui", "limit", "stomach", "yorkkk", "sacrifice", "mon", "resides", "sing", "master", "banks", "dances", "artistic", "ã", "mona", "worrying", "dying", "idk", "multicultural", "years", "fulfilled", "vivid", "favelas", "croissants", "tasting", "first", "active", "prosperityâ", "burden", "proceeded", "vision", "summer", "blessed", "brand", "walk", "paris", "gun", "rome", "person", "bugs", "like", "ice", "intersection", "chosen", "musical", "overwatch", "reason", "gusteau", "activities", "quote", "hard", "bakeries", "black", "bonita", "cities", "aux", "inevitable", "empty", "nocturnal", "stub", "adventure", "skyline", "week", "exploitation", "emirates", "women", "faster", "pizza", "loving", "flashy", "z", "riches", "joanne", "himid", "jai", "tbh", 
"cant", "huge", "fused", "none", "tennis", "written", "hypermodernized", "summers", "wet", "artist", "shall", "dog", "liberty", "model", "sunny", "footballers", "cristo", "crust", "surrounded", "ú", "learn", "moneyâ", "pen", "rotten", "mans", "hear", "bottom", "celebrate", "let", "expectations", "urban", "hub", "ever", "power", "louvre", "decorations", "record", "prosperous", "dirty", "turn", "designed", "find", "realize", "œ", "exercise", "put", "sera", "irving", "jive", "night", "lamborghini", "restart", "ted", "two", "digennaro", "popular", "mountain", "thus", "overrated", "happen", "donc", "iconic", "foie", "digan", "three", "queens", "one", "colorful", "bandolero", "politics", "toronto", "richer", "enduring", "saying", "energy", ".", "size", "lived", "grew", "acquired", "hot", "existence", "round", "technology", "old", "capitalism", "wherever", "beach", "sins", "rains", "effile", "stops", "tate", "jose", "still", "poetry", "wealthy", "plane", "everything", "situations", "springing", "ends", "higher", "slums", "less", "players", "touched", "oh", "powers", "nation", "bad", "frenchman", "festive", "dark", "score", "answer", "university", "vacations", "thoreau", "heightsâ", "professional", "right", "loca", "cold", "thought", "sucked", "work", "game", "appetit", "true", "nightmare", "taylor", "pack", "m", "firm", "cake", "number", "disillusionment", "people", "al", "henry", "downtown", "wannabe", "gold", "petroleum", "pick", "turned", "show", "projects", "gain", "brain", "looking", "samba", "beaches", "alicia", "soccer", "satisfy", "big", "brasil", "vamos", "alright", "feel", "suit", "toâ", "spreadin", "hope", "trash", "seems", "batman", "could", "palm", "smart", "use", "even", "stone", "living", "statue", "viral", "ostriches", "golden", "really", "ur", "magnificent", "chbosky", "jungle", "donde", "christ", "divided", "much", "stocks", "club", "modernityâ", "ronaldoâ", "hell", "souls", "sous", "hands", "united", "afford", "dries", "landmarks", "migrant", "yallah", "grow", "excellence", "loaf", "since", "honeymoon", "feijoada", "cuisine", "feels", "makeâ", "follower", "culture", "celebration", "rest", "built", "mutant", "lindsey", "opprtunity", "always", "stole", "calculated", "king", "smell", "geographical", "exceeded", "america", "brings", "skyscraper", "traffic", "anywhere", "perfect", "fiesta", "landscape", "idea", "voy", "ride", "swimming", "hudson", "las", "la", "infinity", "elegance", "plait", "blu", "possibly", "amie", "appleâ", "bikini", "spread", "trips", "gave", "millions", 
"new", "lisa", "sibling", "glitz", "ronaldo", "drinks", "delicate", "paint", "fries", "stayâ", "marvelous", "rules", "memorable", "tumbrels", "workers", "broke", "freeeedom", "things", "carnivale", "disgusting", "embarrass", "candy", "former", "art", "crime", "busy", "goddamn", "neymarrrr", "man", "race", "ultramodern", "brodsky", "memory", "lovers", "dickensâ", "bailao", "u", "tourists", "location", "lexicon", "industry", "mixed", "fact", "economic", "backed", "silence", "y", "miracle", "progress", "kelk", "pretty", "equality", "el", "enough", "luck", "development", "bed", "passion", "imposssible", "romanticâ", "someone", "anyone", "blank", "balanced", "possibilities", "cost", "tears", "dessert", "sand", "ê", "soul", "luxury", "lies", "within", "time", "vi", "gorgeous", "concrete", "apartment", "feeling", "rude", "finds", "yorrrkk", "liable", "grandmaster", "believe", "antidote", "overly", "desert", "perhaps", "hotspot", "gas", "going", "war", "perfume", "dollars", "sculpture", "lavish", "architectural", "short", "trae", "andrew", "eiffel", "skyscrapers", "rainy", "harsh", "ninja", "mundo", "diet", "eat", "lost", "rhythm", "poorer", "want", "party", "job", "judge", "says", "chef", "comfortable", "color", "de", "haute", "expensive", "taxes", "opulence", "swift", "lama", "state", "laugh", "dorothy", "ow", "revolution", "violations", "grand", "london", "eats", "lights", "croissan", "safety", "surrender", "pele", "reach", "horses", "blood", "reflected", "teaches", "manhattan", "masked", "ici", "barney", "quotes", "code", "later", "wakin", "controls", "feast", "inspire", "dreams", "belongs", "beautiful", "parado", "gon", "wait", "food", "remember", "slave", "specter", "back", "saves", "would", "sleeps", "unconditionally", "imagined", "conservative", "meant", "specific", "zooyork", "sun", "houses", "willing", "maybe", "greatness", "say", "campeo", "style", "cook", "system", "fun", "term", "west", "tunnels", "cristiano", "watercolor", "roll", "finance", "tourist", "bullish", "count", "nights", "visit", "ought", "raised", "junk", "novo", "statueâ", "riding", "carry", "sports", "rose", "classmates", "reality", "stays", "ofâ", "torch", "everywhere", "cristovao", "great", "monsters", "meaningful", "close", "friendship", "gang", "barefoot", "architecture", "vous", "arts", "morning", "islands", "successful", "starts", "coastal", "boundless", "worm", "cigarettes", "hierarchy", "hey", "lower", "purpose", "affordable", "probably", "points", "koch", "loved", "un", "pursuit", "square", 
"rent", "une", "accept", "captured", "loveâ", "surface", "getâ", "amigo", "sunshine", "accountants", "flow", "pony", "mbappe", "jr", "parisâ", "footbal", "greed", "items", "cultures", "wings", "finish", "inclusive", "soy", "full", "grey", "based", "confidently", "destiny", "please", "pockets", "ca", "line", "excellent", "walt", "rural", "human", "make", "tax", "oasis", "facade", "next", "east", "argument", "tomorrow", "land", "wonder", "sound", "country", "poverty", "carrie", "steven", "works", "classic", "sport", "third", "yorkðÿžµ", "enthusiasm", "awesome", "exciting", "yes", "mean", "economically", "lots", "culturally", "aunque", "princes", "jungles", "kylian", "enter", "played", "face", "league", "lets", "founded", "driving", "views", "version", "bradshaw", "gooooaaaaalll", "harris", "parody", "side", "ambitions", "prowess", "olympics", "colors", "businesswomen", "skies", "barely", "crossroads", "immigrants", "na", "developed", "ayy", "psycho", "cool", "artists", "goodbye", "riots", "tropical", "ball", "yeah", "good", "wrapper", "already", "fuel", "deep", "ç", "marvel", "weak", "somehow", "dancing", "history", "present", "divide", "enjoys", "tiny", "escargot", "gathers", "mouth", "theâ", "passionate", "requires", "Î", "hello", "raise", 
"sell", "universe", "siuu", "worthwhile", "cigarettesâ", "paintbrush", "comes", "excessâ", "movie", "symbol", "trading", "heights", "see", "puffy", "appears", "better", "top", "middle", "bon", "subways", 
"jealous", "michael", "yellow", "seaside", "cards", "ashamed", "universes", "redeemer", "tall", "ere", "despair", "mosby", "lincoln", "almost", "rely", "canâ", "breathtaking", "succeed", "problems", "center", "dictatorship", "fifa", "sku", "business", "france", "zoo", "superpowers", "futball", "dubai", "building", "preservences", "something", "income", "word", "lasts", "meet", "extreme", "ho", "element", "somewhat", 
"happens", "source", "suis", "vie", "wall", "dedication", "testament", "quit", "serves", "landscapes", "gigantic", "fraternity", "pot", "head", "minutes", "aller", "subway", "similarities", "loud", "class", "luxurious", "brokerage", "pense", "challenges", "insist", "jackson", "times", "janeiroâ", "pain", "windy", "futbol", "cultural", "way", "felt", "ugliness", "err", "stepping", "skyscrapersâ", "anything", "marley", "science", "new york", "dine", "ammazing", "eyes", "infrastructure", "american", "neymar", "paradono", "souffle", "along", "tell", "steve", "turtles", "arab", "question", "thinking", "flags", "today", "exchange", "en", "need", "pay", "news", "oinion", "experienced", "continuous", "erase", "experience", "choice", "every", "another", "constantly", "never", "mere", "live", "keeps", "made", "lucky", "distinguishes", "heads", "sriram", "happiness", "many", "stupid", "cup", "ford", "die", "é", "despite", "routine", "street", "nyc", "historical", "writes", "sandy", "twice", "enjoyed", "name", "letter", "berlin", "timeless", "court", "antiquated", "thousand", "cover", "fashion", "exploit", "fast", "blue", "diversity", "succumb", "deal", "innovation", "us", "est", "abundant", "cozy", "fulfilment", "places", "alive", "rhythms", "kid", "held", "harder", "peux", "naturalistic", "states", "lazy", "keller", "dj", "greatest", "metropolitan", "playground", "bird", "respectâ", "brazil", "average", "mark", "little", "fossil", "young", "leroux", "carnivals", "leavin", "look", "taken", "connect", "suiiiiiiiiiiiii", "baguetteâ", "takes", "know", "must", "named", "metropolis", "illegal", "gras", "prince", "seasons", "cop", "martial", "h", "treasure", "guess", "variety", "walking", "started", "warm", "freedom", "life", "arabic", "filled", "sorry", "shallow", "loss", "useless", "famous", "peter", "je", "away", "large", "stealing", "copious", "language", "tourism", "natural", "direction", "noisy", "carnival", "decisions", "however", "businessmen", "crowded", "days", "vogue", "fight", "license", "rather", "absolutely", "salute", "tech", "companies", "roof", "floating", "redentor", "carnaval", "play", "que", "coffee", "tradition", "rats", "turns", "get", "wake", "city", "rio", "year", "becoming", "bailar", "allons", "matuidi", "also", "richest", "nature", "imagination", "carefree", "baby", "mi", "fear", "either", "yorkers", "setback", "furious", "soup", "nice", "alot", "guillotine", "parties", "stories", "responsibility", "cheese", "deserves", "criticism", "born", "im", "end", "serenity", "K", "future", "wildest", "economy", "angery", "redemption", "six", "è", "probs", "thereafter", "centre", "ny", "absolute", "area", "exist", "vacation", "vibrant", "partying", "sings", "story", "buy", "futuristic", "care", "carried", "influencers", "climate", "bigâ", "worry", "bolder", "property", "yalla", "found", "joseph", "stinson", "trap", "simple", "soulless", "ambition", "clouds", "actually", "artificial", "familia", "obvious", "oui", "wish", 
"colette", "guests", "vegas", "colourful", "jesus", "think", "well", "moveable", "sands", "route", "soccerâ", "football", "t", "modern", "course", "taking", "sky", "imagine", "whispers", "patrick", "festivals", "welcome", "part", "heaven", "coolest", "melting", "century", "celebrations", "ratatouille", "cheaper", "family", "inequality", "wilde", "ba", "olympic", "moments", "homes", "empire", "lines", 
"deserve", "rumble", "ocean", "bob", "romance", "kansas", "oil", "influential", "looooveâ", "annual", "strength", "palaces", "senses", "du", "understanding", "health", "general", "happy", "meets", "buildings", 
"marvels", "happened", "worst", "fearless", "silver", "roots", "lose", "entire", "thrives", "prosperity", "panda", "butterflies", "vive", "toto", "relentless", "spirit", "finite", "upon", "richâ", "forest", "angry", "anxious", "amounts", "secrets", "without", "intouchablesâ", "clears", "sleepless", "sao", "romans", "come", "poor", "father", "apart", "failure", "shining", "chaotic", "waves", "ô", "homeâ", "piece", "yet", "scene", "alone"
"bonsoir", "bonjour", "non", "excusez-moi", "je", "suis", "désolé", "s'il", "vous", "plaît", "merci", "notre", "dame", "cathedral", "palais" "garnier", "palais", "olá", "habibi", "bom", "dia", "allah", "por", "olympic"]

city_to_index = {
    "Dubai": 0,
    "Paris": 1,
    "New York City": 2,
    "Rio de Janeiro": 3
}

def make_bow(data, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
        `t`: A numpy array of shape [len(data)], with `t[i] == 1` if
             `data[i]` is a positive review, and `t[i] == 0` otherwise.
    """
    X = np.zeros([len(data), len(vocab)])
    t = np.zeros([len(data)])
    
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    # TODO: fill in the appropriate values of X and t
    # It enumerates data then loops vocab so it takes data * vocab proof by brain
    for i, (quote, city) in enumerate(data):
        t[i] = city_to_index[city]
        toString = str(quote)
        words = set(toString.lower().split()) 

        for word in words:
            if(word in vocab_index):
                j = vocab_index[word]
                X[i, j] = 1

    return X, t

def naive_bayes_mle(X, t):
    """
    Compute the parameters $pi$ and $theta_{jc}$ that maximizes the log-likelihood
    of the provided data (X, t).

    **Your solution should be vectorized, and contain no loops**

    Parameters:
        `X` - a matrix of bag-of-word features of shape [N, V],
              where N is the number of data points and V is the vocabulary size.
              X[i,j] should be either 0 or 1. Produced by the make_bow() function.
        `t` - a vector of class labels of shape [N], with t[i] being either 0 or 1.
              Produced by the make_bow() function.

    Returns:
        `pi` - a scalar; the MLE estimate of the parameter $\pi = p(c = 1)$
        `theta` - a matrix of shape [V, 2], where `theta[j, c]` corresponds to
                  the MLE estimate of the parameter $\theta_{jc} = p(x_j = 1 | c)$
    """
    N, V = X.shape
    C = len(np.unique(t))

    # Calculate class priors based on frequency
    pi_mle = np.zeros(C)
    for c in range(C):
        pi_mle[c] = np.sum(t == c) / N

    # Calculate conditional probabilities for each feature given the class
    theta_mle = np.zeros((V, C))
    for c in range(C):
        X_c = X[t == c]
        # Applying Laplace smoothing to avoid zero probabilities for unseen features
        theta_mle[:, c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)

    return pi_mle, theta_mle

def make_prediction(X, pi, theta):
    log_theta = np.log(theta)
    log_theta_neg = np.log(1 - theta)
    class_log_probs = X.dot(log_theta) + (1 - X).dot(log_theta_neg)
    class_log_probs += np.log(pi)
    y = np.argmax(class_log_probs, axis=1)

    return y

# Vectorization
#
#
#
#
#
#
def process_Q10(data):
    categories_q5 = ["Dubai", "Paris", "New_York", "Rio"]
    categories = ["0", "1", "2", "3"]
    for category in categories:
        data[f"Q10_{categories_q5[int(category)]}"] = (data["newQ10"] == int(category)).astype(int)

    categories_q5 = ["Dubai", "Paris", "New_York", "Rio"]

    del data["newQ10"]

def process_Q56(data):
    categories = ["Friends", "Co-worker", "Siblings", "Partner"]
    for category in categories:
        data[f"Q5_{category}"] = (data["Q5"] == category).astype(int)

    categories_q5 = ["Friends", "Co-worker", "Sibling", "Partner"]
    categories_q6 = ["Skyscrapers", "Sport", "Art and Music", "Carnival", "Cuisine", "Economic"]
    patterns = {category: f"{category}=>(\d+)" for category in categories_q6}

    for category in categories_q5:
        data[f"Q5_{category}"] = data["Q5"].str.contains(category, na=False).astype(int)

    del data["Q5"]
    
    for category, pattern in patterns.items():
        data[category] = data["Q6"].str.extract(pattern).astype(float).fillna(0)

    del data["Q6"]

def clean_up_input(data):
    data["Q7"] = data["Q7"].apply(to_numeric).fillna(0)
    data["Q8"] = data["Q8"].apply(to_numeric).fillna(0)
    data["Q9"] = data["Q9"].apply(to_numeric).fillna(0)

    # Clean for number categories
    data["Q1"] = data["Q1"].apply(get_number)
    data["Q2"] = data["Q2"].apply(get_number)
    data["Q3"] = data["Q3"].apply(get_number)
    data["Q4"] = data["Q4"].apply(get_number)

    return data

def evaluate_alpha(alpha, X_train, y_train, X_valid, y_valid, n_runs=100):
    accuracies = []
    for _ in range(n_runs):
        w = solve_via_sgd(X_train, y_train, alpha=alpha, n_epochs=40, batch_size=50)
        accuracy_val = accuracy(w, X_valid, y_valid)
        accuracies.append(accuracy_val)
    return np.mean(accuracies)

def find_best_alpha(X_train, y_train, X_valid, y_valid, alphas, n_runs=100):
    best_alpha = None
    best_accuracy = -1

    for alpha in alphas:
        print(f"Evaluating alpha = {alpha}")
        avg_accuracy = evaluate_alpha(alpha, X_train, y_train, X_valid, y_valid, n_runs=n_runs)
        print(f"Average Validation Accuracy for alpha = {alpha}: {avg_accuracy:.4f}")

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_alpha = alpha
    
    return best_alpha, best_accuracy

if __name__ == "__main__":
    #Vectorizeation
    np.set_printoptions(threshold=np.inf)
    data = clean_up_input(data)
    process_Q56(data)

    # Look at unsupervised cluster.py It is now a MLE, but i did not change the name. I tried K-Clustering like lab 10 but did not work so did lab 9 instead with MLE.
    # Reason I choose MLE over MAP is because I do not know the prior probability of a city. It could be 1/4 every time, or the data set can be evil and be 100% one city. Who knows
    newData = list(zip(data["Q10"], data["Label"]))
    X_train, t_train = make_bow(newData, vocab)
    pi_mle, theta_mle = naive_bayes_mle(X_train, t_train)
    y_mle_train = make_prediction(X_train, pi_mle, theta_mle)
    predictions_and_actuals = list(zip(y_mle_train, data["Label"]))
    data["newQ10"] = y_mle_train
    
    process_Q10(data)

    # Here Vincent put all the vectors. Q1-4 and Q6 were not binary classification because they have a rank relationship, binary classification will not caputure that idea
    # The MLE prediction and Q5 was vectorized binary because it is either present that the model predicted a city or present someone will bring their freinds or it is not
    # Only Q7-Q9 are normalized because only those questions are continous(user input) and large. They need to be normalized
    # There is a bias just like lab
    data_fets = np.stack([
        np.ones((data.shape[0])),
        (data["Q1"]),
        (data["Q2"]),
        (data["Q3"]),
        (data["Q4"]),
        data["Q5_Friends"],
        data["Q5_Co-worker"],
        data["Q5_Siblings"],
        data["Q5_Partner"],
        data["Skyscrapers"],
        data["Sport"],
        data["Art and Music"],
        data["Carnival"],
        data["Cuisine"],
        data["Economic"],
        data["Q10_Dubai"],
        data["Q10_Paris"],
        data["Q10_New_York"],
        data["Q10_Rio"],
        #Regression here sort of, numbers are continous / boundless
        (data["Q7"]),
        (data["Q8"]),
        (data["Q9"]),
    ], axis=1)
    numerical_value_start = 19

    #Creating target and features mapping
    X = data_fets

    label_mapping = {
        "Dubai": 0,
        "Rio de Janeiro": 1,
        "New York City": 2,
        "Paris": 3
    }

    t = data["Label"].map(label_mapping)
    X_train, X_valid, t_train, t_valid = train_test_split(X, t, train_size=869/1469)
    numerical_value_start = 5

    # The mean/std of the numerical features over X_train
    # The normalizing like lab 5
    mean = X_train[:, numerical_value_start:].mean(axis=0)
    std = X_train[:, numerical_value_start:].std(axis=0)
    
    X_train_norm = X_train.copy()
    X_valid_norm = X_valid.copy()
        
    X_train_norm[:, numerical_value_start:] = (X_train[:, numerical_value_start:] - mean) / std
    X_valid_norm[:, numerical_value_start:] = (X_valid[:, numerical_value_start:] - mean) / std
    w = solve_via_sgd(alpha=0.05, X_train=X_train_norm, t_train=t_train, n_epochs=40, batch_size=50)
    y_pred = pred(X_valid_norm, w)

    
    alphas = [0.001, 0.075, 0.025, 0.01, 0.05, 0.1, 0.5]

    validation_accuracy = accuracy(w, X_valid_norm, t_valid)
    best_alpha, best_accuracy = find_best_alpha(X_train_norm, t_train, X_valid_norm, t_valid, alphas, n_runs=100)

    print(f"Best alpha: {best_alpha} with average accuracy: {best_accuracy:.4f}")


# numpy and pandas are also permitted
import numpy as np
import pandas as pd

# basic python imports are permitted
import sys
import csv
import re
import random

data = pd.read_csv("clean_dataset.csv")
newData = []
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

def make_bow(quotes, vocab):
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
    X = np.zeros([len(quotes), len(vocab)])
    t = np.zeros([len(quotes)])
    
    vocab_index = {word: idx for idx, word in enumerate(vocab)}

    # TODO: fill in the appropriate values of X and t
    # It enumerates data then loops vocab so it takes data * vocab proof by brain
    for i, (quote, city) in enumerate(quotes):
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

    pi_mle = np.zeros(C)
    for c in range(C):
        pi_mle[c] = np.sum(t == c) / N

    theta_mle = np.zeros((V, C))
    for c in range(C):
        X_c = X[t == c]
        theta_mle[:, c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)

    return pi_mle, theta_mle

#Printing stuff
def make_prediction(X, pi, theta):
    log_theta = np.log(theta)
    log_theta_neg = np.log(1 - theta)
    class_log_probs = X.dot(log_theta) + (1 - X).dot(log_theta_neg)
    class_log_probs += np.log(pi)
    y = np.argmax(class_log_probs, axis=1)

    return y

def accuracy(y, t):
    return np.mean(y == t)

def count_city_predictions(predictions):
    
    dubai = {"0": 0, "1": 0, "2": 0, "3": 0}
    new = {"0": 0, "1": 0, "2": 0, "3": 0}
    paris = {"0": 0, "1": 0, "2": 0, "3": 0}
    rio = {"0": 0, "1": 0, "2": 0, "3": 0}
    newlist = [dubai, paris, new, rio]

    for prediction, number in predictions:
        city = city_to_index[number] 
        newlist[city][str(prediction)] += 1

    return newlist


if __name__ == "__main__":
    #Deleted all the binary vectors of q1-q4 impossible to work with
    #Vectorizeation
    
    newData = list(zip(data["Q10"], data["Label"]))
    X_train, t_train = make_bow(newData, vocab)
    pi_mle, theta_mle = naive_bayes_mle(X_train, t_train)
    print(pi_mle.shape, theta_mle.shape)
    y_mle_train = make_prediction(X_train, pi_mle, theta_mle)
    
    np.set_printoptions(threshold=np.inf)
    print("MLE Train Acc:", accuracy(y_mle_train, t_train))
    predictions_and_actuals = list(zip(y_mle_train, data["Label"]))
    #print(predictions_and_actuals)
    data["newQ10"] = y_mle_train
    print(data["newQ10"])

    #print(count_city_predictions(predictions_and_actuals))




# numpy and pandas are also permitted
import numpy as np
import pandas as pd

# basic python imports are permitted
import sys
import csv
import re
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

data = pd.read_csv("clean_dataset.csv")
newData = []
vocab = set()

#LAB 10 print
def m_step(X, R):
    """
    Perform for a BMM the "M" step of the E-M Algorithm for a BMM.
    In other words, given the data matrix `X` and estimates to the
    parameters `theta` and `pi`, estimate $P(z=k|{\bf x})$ for each ${\bf x}$ in the
    data matrix.

    Parameters:
        `X` - a data matrix of bag-of-word features of shape [N, D],
              where N is the number of data points and D is the vocabulary size.
              X[i,j] should be either 0 or 1. Produced by the make_bow() function.
        `R` - a matrix responsibilities of shape [N, K], where `R[j, k]` corresponds
              to the value $P(z^{(j)}=k| {\bf x^{(j)}})$ computed during the e_step.
              Precondition: Each row of `R` sums to 1, i.e. `sum(R[j,:]) == 1`

    Returns:
        `theta` - a matrix of shape [D, K], where `theta[j, k]` corresponds to
              the MLE estimate of the parameter $\theta_{jk} = p(x_j = 1 | z=k)$
        `pi` - a vector of shape [K], where `pi[k]` corresponds to
              the MLE estimate of the parameter $\pi_{k} = P(z=k)$.
              We should have `np.sum(pi) = 1` so that the $\pi_k$s describe a
              probability distribution.
    """
    N, D = X.shape
    N, K = R.shape

    pi = np.sum(R, axis=0) / N # todo
    theta = np.zeros((D, K)) # TODO: fill this!
    for k in range(K):
        for j in range(D):
            theta[j, k] = np.dot(R[:, k], X[:, j]) / np.sum(R[:, k])

    return pi, theta

if __name__ == "__main__":
    #Deleted all the binary vectors of q1-q4 impossible to work with
    #Vectorizeation
    
    newData = data["Q10"]

    nltk.download("punkt")  # For tokenization

    # Download stop words
    nltk.download("stopwords") 

    
    stop_words = set(stopwords.words("english"))

    def remove_stopwords(sentence):
        sentence = str(sentence)
        tokens = word_tokenize(sentence.lower())  # Tokenize and lowercase
        filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]  # Remove stop words and punctuation
        return " ".join(filtered_tokens)

    # Apply the function to the Q10 column
    data["Q10_filtered"] = data["Q10"].apply(remove_stopwords)

    dubai = ["Burj Khalifa", "Palm Jumeirah", "Dubai Mall", "Luxury", "Skyscrapers", "Desert", "Innovation", "Expo", "Marina", "Emirates", "Gold Souk", "Futuristic", "Tourism", "Arabian", "Wealth", "Business Hub", "Safari", "Artificial Islands", "Shopping", "Finance", "Skyline", "Oil", "Global City", "Luxury Shopping", "Hospitality", "International", "Creek", "Culture", "Trade", "Infrastructure", "Metro", "Futuristic Architecture", "Real Estate", "Investment", "Dhow", "Beaches", "Resort", "Nightlife", "Culinary", "Festival", "Heritage", "Museum", "Aquarium", "Waterpark", "Adventure", "Golf", "Sports", "Aviation", "Technology", "Environment", "Sustainability", "Solar Energy", "Vertical Farming", "Robotics", "Artificial Intelligence", "Smart City", "Drones", "3D Printing", "Autonomous Vehicles", "Blockchain", "Cybersecurity", "Digital Economy", "E-Government", "Fintech", "Green Building", "Hydroponics", "Internet of Things", "Logistics", "Renewable Energy", "Space Exploration", "Telecommunication", "Urban Planning", "Virtual Reality", "Augmented Reality", "Waste Management", "Water Desalination", "Wind Energy", "Youth Empowerment", "Zero Emissions", "High-Speed Rail", "Cloud Computing", "Data Analytics", "E-Commerce", "Electric Vehicles", "Gamification", "HealthTech", "Industrial Automation", "Machine Learning", "Nanotechnology", "Quantum Computing", "RegTech", "Social Media", "Startup Ecosystem", "User Experience", "Venture Capital", "Wearable Technology", "5G Network", "Biotechnology", "CleanTech", "Digital Marketing", "EdTech", "FoodTech", "Genetic Engineering", "Holography", "Immersive Technology", "Judicial Tech", "Knowledge Economy", "LegalTech", "Mobile Apps", "Neuroscience", "Online Education", "Precision Medicine", "Quantum Encryption", "Robotics Engineering", "Satellite Technology", "Thermal Energy", "Unmanned Aerial Vehicles", "Virtual Assistant", "Web Development", "Xenotransplantation", "Yachting Industry", "Zeolite Technology", "World Expo", "Al Fahidi Fort", "Dubai Opera", "Jumeirah Mosque", "Al Bastakiya", "Dubai Frame", "Palm Jabel Ali", "Deira", "Dubai Creek Harbor", "Bluewaters Island", "Ain Dubai", "Dubai Canal", "Dubai Marina Yacht Club", "Madinat Jumeirah", "Dubai Sports City", "Dubai Silicon Oasis", "Al Maktoum International Airport"]
    paris = ["Eiffel Tower", "Louvre Museum", "Notre Dame", "Champs-Elysées", "Seine River", "Montmartre", "Sacré-Cœur", "Luxembourg Gardens", "Sainte-Chapelle", "Latin Quarter", "Moulin Rouge", "Fashion", "Cuisine", "Art", "Culture", "History", "Romance", "Cafés", "Croissant", "Baguette", "Wine", "Cheese", "Haute Couture", "Impressionism", "Literature", "Philosophy", "Gothic Architecture", "Street Art", "Ballet", "Opera", "Jazz", "Film", "Le Marais", "Versailles", "Patisserie", "Macaron", "Escargot", "Foie Gras", "Coq au Vin", "Ratatouille", "Crepes", "Tarte Tatin", "Champagne", "Bordeaux", "Sorbonne", "Bouquinistes", "Rue Cler", "Père Lachaise Cemetery", "Orsay Museum", "Rodin Museum", "Picasso Museum", "Quiche", "Madeleine", "Clafoutis", "Bouillabaisse", "Camembert", "Brie", "Roquefort", "Fashion Week", "L'Oréal", "Chanel", "Dior", "Yves Saint Laurent", "Louis Vuitton", "Givenchy", "Hermès", "Lanvin", "Jean Paul Gaultier", "Pierre Cardin", "Nina Ricci", "Champs de Mars", "Tuileries Garden", "Place de la Concorde", "Arc de Triomphe", "Pont Neuf", "Île de la Cité", "Île Saint-Louis", "Sorbonne University", "Pantheon", "Les Invalides", "Opera Garnier", "Palais Royal", "Grand Palais", "Petit Palais", "Place Vendôme", "Place des Vosges", "Saint Germain des Prés", "Montparnasse", "La Défense", "Gobelins", "Belleville", "Canal Saint-Martin", "Place de la Bastille", "Gare du Nord", "Gare de Lyon", "Gare Saint-Lazare", "Orly Airport", "Charles de Gaulle Airport", "Bateaux Mouches", "Rue Montorgueil", "Marché des Enfants Rouges", "Galeries Lafayette", "Printemps", "Le Bon Marché", "Saint-Ouen Flea Market", "Carnavalet Museum", "Jacquemart-André Museum", "Cluny Museum", "Grévin Museum", "Paris Plages", "Nuit Blanche", "Fête de la Musique", "Paris Marathon", "Tour de France", "French Open", "Fashion Capital", "City of Light", "City of Love", "Luxembourg Palace", "Conciergerie", "Palais de Tokyo", "Centre Pompidou", "Musée de l'Orangerie", "Musée Marmottan Monet", "Fondation Louis Vuitton", "Parc des Princes", "Stade de France", "Le Zénith", "Philharmonie de Paris", "La Cinémathèque Française", "Le Grand Rex", "Shakespeare and Company", "Berthillon Ice Cream", "Angelina Tea House", "Café de Flore", "Les Deux Magots", "Brasserie Lipp", "La Coupole", "Le Procope", "La Closerie des Lilas", "Le Select", "Café des Deux Moulins", "Café de la Paix", "Maxim's", "Ladurée", "Fauchon", "Pierre Hermé", "Maison Kayser", "Poilâne", "BHV Marais", "Forum des Halles", "Le Marais", "Bastille Day", "Mona Lisa", "Venus de Milo", "Winged Victory of Samothrace", "Liberty Leading the People", "Gargoyles", "French Revolution", "Napoleon Bonaparte", "Sun King", "Gilles Clément", "Pierre de Ronsard", "Édith Piaf", "Gertrude Stein", "Ernest Hemingway", "F. Scott Fitzgerald", "Oscar Wilde", "Victor Hugo", "Alexandre Dumas", "Marcel Proust", "Simone de Beauvoir"]
    new_york = ["Statue of Liberty", "Empire State Building", "Central Park", "Times Square", "Brooklyn Bridge", "Metropolitan Museum of Art", "Broadway", "Wall Street", "United Nations", "5th Avenue", "Rockefeller Center", "One World Trade Center", "The High Line", "Museum of Modern Art", "New York Public Library", "Grand Central Terminal", "Chinatown", "Little Italy", "Harlem", "Greenwich Village", "SoHo", "Tribeca", "Lower East Side", "East Village", "West Village", "Midtown Manhattan", "Upper East Side", "Upper West Side", "Battery Park", "Ellis Island", "Guggenheim Museum", "American Museum of Natural History", "Chelsea Market", "Central Park Zoo", "Hudson River", "East River", "Bronx Zoo", "Coney Island", "Staten Island Ferry", "New York Stock Exchange", "NASDAQ", "Carnegie Hall", "Madison Square Garden", "Yankee Stadium", "Mets", "Knicks", "Rangers", "Giants", "Jets", "Brooklyn Nets", "New York City Marathon", "New York Fashion Week", "The New Yorker", "Times", "Saturday Night Live", "Wall Street Journal", "New York Times", "Vogue", "Met Gala", "Tiffany & Co", "Bloomingdale's", "Macy's Thanksgiving Day Parade", "New Year's Eve Ball Drop", "Saks Fifth Avenue", "Barney's", "FAO Schwarz", "Bergdorf Goodman", "The Plaza Hotel", "Waldorf Astoria", "Ritz-Carlton", "Four Seasons", "Luxury", "Shopping", "Fashion", "Arts", "Theater", "Cuisine", "Diversity", "Innovation", "Architecture", "Skyscrapers", "History", "Culture", "Subway", "Taxi", "Boroughs", "Neighborhoods", "Parks", "Recreation", "Education", "Finance", "Media", "Entertainment", "Sports", "Tourism", "Landmarks", "Icons", "Heritage", "Festivals", "Museums", "Galleries", "Exhibitions", "Concerts", "Opera", "Ballet", "Jazz", "Hip-Hop", "Rock", "Pop", "Classical", "Literature", "Poetry", "Film", "Television", "Comedy", "Drama", "Adventure", "Nightlife", "Bars", "Clubs", "Restaurants", "Cafes", "Bakeries", "Delis", "Street Food", "International Cuisine", "Pizza", "Bagels", "Hot Dogs", "Cheesecake", "Coffee", "Craft Beer", "Cocktails", "Wine", "Markets", "Shopping Districts", "Boutiques", "Designer Stores", "Vintage", "Antiques", "Artisan", "Craftsmanship", "Technology", "Startups", "Business", "Economy", "Healthcare", "Wellness", "Fitness", "Yoga", "Pilates", "Running", "Cycling", "Hiking", "Swimming", "Boating", "Fishing", "Gardening", "Photography", "Drawing", "Painting", "Sculpture", "Ceramics", "Printmaking", "Design", "Fashion Design", "Graphic Design", "Interior Design", "Landscape Design", "Urban Design", "Education", "Research", "Science", "Mathematics", "Engineering", "Technology", "Innovation", "Entrepreneurship", "Leadership", "Management", "Marketing", "Economics", "Accounting", "Finance", "Law", "Politics", "Government", "Public Policy", "Community Service", "Volunteering", "Philanthropy", "Environmentalism", "Sustainability", "Renewable Energy", "Conservation", "Wildlife", "Flora", "Fauna", "Climate Change", "Global Warming", "Pollution", "Recycling", "Waste Management", "Water Resources", "Air Quality", "Soil Health", "Agriculture", "Forestry", "Fishing", "Mining", "Manufacturing", "Construction", "Transportation", "Logistics", "Warehousing", "Retail", "Wholesale", "International Trade", "Exports", "Imports", "Globalization", "Cultural Exchange", "Diplomacy", "International Relations", "Peace", "Conflict Resolution", "Human Rights", "Justice", "Equality", "Diversity", "Inclusion", "Accessibility", "Health",]
    rio = ["Christ the Redeemer", "Sugarloaf Mountain", "Copacabana Beach", "Ipanema Beach", "Carnaval", "Samba", "Bossanova", "Maracanã Stadium", "Tijuca National Park", "Rodrigo de Freitas Lagoon", "Selarón Steps", "Rio Carnival", "Museum of Tomorrow", "São Sebastião Cathedral", "Santa Teresa", "Lapa Arches", "Botanical Garden", "Flamengo Park", "Corcovado Mountain", "Pedra da Gávea", "Prainha Beach", "Grumari Beach", "Arpoador", "Favelas", "Carioca", "Feijoada", "Caipirinha", "Churrascaria", "Açaí", "Pão de Queijo", "Brigadeiro", "Cachaça", "Soccer", "Beach Volleyball", "Surfing", "Hang Gliding", "Paragliding", "Cycling", "Jogging", "Capoeira", "Street Art", "Cultural Festivals", "Music Festivals", "Film Festival", "Art Galleries", "Historic Architecture", "Colonial Churches", "Modernist Buildings", "Oscar Niemeyer", "Porto Maravilha", "Pedro do Sal", "Imperial Palace", "National Library", "Rio Zoo", "AquaRio", "Maracanãzinho", "Sambadrome Marquês de Sapucaí", "Cinelândia", "Catete Palace", "Municipal Theater", "National Museum of Fine Arts", "Museum of Modern Art Rio", "Museum of Contemporary Art", "Ilha Fiscal", "Quinta da Boa Vista", "Fort Copacabana", "Sugarloaf Cable Car", "Urca", "Guanabara Bay", "Barra da Tijuca", "Leblon", "Gávea", "Jardim Botânico", "Lagoa", "Recreio dos Bandeirantes", "Angra dos Reis", "Paquetá Island", "Cagarras Islands", "Petrobras", "Vale", "Carnival Parade", "Samba Schools", "Bloco", "Sambódromo", "Feira de São Cristóvão", "Portuguese Tile", "Igreja da Candelária", "Monastery of São Bento", "Ilha de Paquetá", "Rio Antigo", "Praça Mauá", "Praça XV", "Escadaria Selarón", "Arcos da Lapa", "Bonde de Santa Teresa", "Parque Lage", "Praia do Pepino", "Praia Vermelha", "Morro Dois Irmãos", "Pedra Bonita", "Floresta da Tijuca", "Mirante Dona Marta", "Pico da Tijuca", "Vista Chinesa", "Mureta da Urca", "Praia do Secreto", "Prainha Branca", "Trilha Transcarioca", "Ciclovia Tim Maia", "Boulevard Olímpico", "Mural Etnias", "Kobra", "Cidade das Artes", "Rio Design Center", "Barra Shopping", "São Conrado Fashion Mall", "Rio Open", "Jeunesse Arena", "HSBC Arena", "Maria Lenk Aquatic Center", "Olympic Golf Course", "Olympic Park", "Rock in Rio", "Theatro Municipal", "Casa de Rui Barbosa", "Instituto Moreira Salles", "Centro Cultural Banco do Brasil", "Oi Futuro", "Casa França-Brasil", "Fundação Eva Klabin", "Fundação Getulio Vargas", "Instituto de Pesquisa Jardim Botânico", "Observatório Nacional", "Parque Nacional da Tijuca", "UNIRIO", "UFRJ", "PUC-Rio", "FGV", "Centro de Arte Helio Oiticica", "Espaço Cultural da Marinha", "Museu da República", "Museu do Amanhã", "Museu Nacional", "Biblioteca Nacional", "Jockey Club Brasileiro", "Lagoa Rodrigo de Freitas", "Baía de Guanabara", "Praia da Joatinga", "Rio Scenarium", "Lapamaki", "Bar do Mineiro", "Aprazível", "Confeitaria Colombo", "Beco das Garrafas", "Jobi", "Botequim Informal", "Academia da Cachaça", "Belmonte", "Bibi Sucos", "Zazá Bistrô", "Azumi", "Miam Miam", "Olympe", "Lasai", "Mee", "Oro", "Le Pré Catelan", "Satyricon", "Gero", "Fasano al Mare", "Pérgula", "Sawasdee", "CT Boucherie", "TT Burger", "Balada Mix", "Via Sete", "Quadrucci", "Braseiro da Gávea", "Guimas", "Galeto Sat's", "Carretão", "Churrascaria Palace", "Mariu's Degustare", "Laguna", "Al Mare", "Bar Lagoa", "Bar Urca", "Pavão Azul", "Adega Pérola", "Cervantes", "Bar da Frente", "Bar da Gema", "Canastra Bar", "Canastra Rosa", "Empório Jardim", "Joaquina", "Venga!", "Void", "Yumê", "Zuka"]
    vocab = set()

    for setence in data['Q10_filtered']:
        word = setence.split()
        for newword in word:
            vocab.add(newword)

    for word in dubai:
        for newword in word:
            vocab.add(newword)

    for word in paris:
        for newword in word:
            vocab.add(newword)

    for word in new_york:
        for newword in word:
            vocab.add(newword)

    for word in rio:
        vocab.add(newword)      


    vocab = list(vocab) # len(vocab) == 1000
    print(vocab)



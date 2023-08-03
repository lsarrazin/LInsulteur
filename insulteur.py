#!/usr/bin/python3

import sys
import os
import time

import random

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import speech_recognition as sr 
from gtts import gTTS

# Configuration de la reconnaissance de voix 
mic_name = "default"
device_id = 14
sample_rate = 48000
chunk_size = 2048

audio = True

# Configuration du nltk
stop_words = set(stopwords.words('french'))


gros_mots_masculin = [
    'Abruti','Ahuri','Analphabète','Anus De Poulpe','Aspirateur A Muscadet','Asticot','Attardé','Avorton',
    'Bachibouzouk','Balai de Chiottes','Barjot','Batârd','Bigleux','Blaireau','Boloss','Bordel à Cul','Boudin','Bouffon','Bougre D’âne',
    'Bougre D’imbécile','Boulet','Bouricot','Boutonneux','Branleur de mouches','Branquignole','Brigand','Butor',
    'Cageot','Canaillou','Cancrelat','Carburateur à Beaujolais','Casse-couilles','Casse-pieds','Cassos','Chacal','Chameau à une bosse','Chancre',
    'Chenapan','Chiassard','Chieur','Clampin','Cloaque','Clodo','Cloporte','Clown','Cochon','Con','Conchieur','Concombre','Connard','Corniaud',
    'Cornichon','Couard','Couillon','Crapaud De Pissotière','Crassard','Crevard','Crâne D’obus','Crétin','Crétin Des Alpes','Crétin Des Iles',
    'Crétin Goîtreux','Cul De Babouin','Cul Terreux',
    'Don Juan De Pissotière','Ducon','Décamerde','Dégueulis','Dégénéré Chromozomique','Dégénéré Du Bulbe','Détritus',
    'Ectoplasme','Emmerdeur','Empaffé','Enculeur de mouches','Enculé','Enfoiré','Eunuque',
    'Faux Jeton','Filou','Fion','Fonctionnaire','Fouille Merde','Four à Merde','Foutriquet','Freluquet','Frippon','Fumier','Furoncle','Félon',
    'Garage A Bite','Gibier De Potence','Gland','Glandeur','Glandu','Gnome','Gogol','Goinfre','Gougnafier','Goujat','Grand Cornichon',
    'Grand Dépandeur D’andouilles','Gras Du Bide','Gredin','Gringalet','Gros Caca Poilu','Gros Con','Gros Lard','Gueux','Gugus','Guignol',
    'Hérétique','Histrion','Hurluberlu',
    'Iconoclaste','Idiot','Ignare','Imbécile','Ivrogne',
    'Jean-foutre','Jobard','Judas',
    'Kéké',
    'Laideron','Lépreux','Lèche-cul',
    'Malandrin','Malotru','Malpropre','Manant','Manche à Couille','Mange Merde','Maquereau','Maraud','Marchand De Tapis','Margoulin','Merdaillon',
    'Merdophile','Merlan Frit','Minus','Moins Que Rien','Molasson','Mongol','Morfale','Morpion','Morveux','Mou Du Bulbe','Mou Du Genou','Mou Du Gland',
    'Mou de la bite','Moule à Gauffre','Mouton De Panurge','Mécréant',
    'Nabot','Nain De Jardin','Nazillon','Necropédophile','Neuneu','Nigaud','Noob','Nécrophile',
    'Obsédé','Oiseau De Mauvaise Augure','Olibrius',
    'Pachyderme','Paltoquet','Panaris','Parasite','Parvenu','Paumé','Pauvre Con','Peigne-cul','Peine à Jouir','Pendard','Pervers',
    'Pet De Moule','Pigeon','Pignouf','Pisse Froid','Pisse-vinaigre','Playboy De Superette','Pleutre','Plouc','Poivrot','Polisson','Poltron','Porc',
    'Pot de chambre','Pouacre','Pourceau','Pousse Mégot','Péquenot','Pétochard',
    'Quadrizomique','Queutard',
    'Radin','Ramassis De Chiure De Moineau','Rambo De Pacotille','Renégat','Roquet','Roublard','Résidu De Fausse Couche','Résidu De Partouze',
    'Sac à Foutre','Sac à Gnole','Sac à Merde','Sac à Viande','Sac à Vin','Sacripan','Sagouin','Salaud','Saligaud','Salopard','Salopiaud',
    'Scaphandrier D’eau De Vaiselle','Scatophile','Scélérat','Schpountz','Sodomite','Sombre Crétin','Sot','Spermatozoide Avarié','Spermiducte',
    'Tâcheron','Tas De Saindoux','Thon','Tire Couilles','Tocard','Tonnerre De Brest','Toqué','Traîne Savate',
    'Tricard','Tromblon','Trou De Balle','Trou Du Cul','Troubignole','Truand','Trumeaux','Tuberculeux','Tudieu','Tétârd',
    'Usurpateur',
    'Va Nu Pieds','Vandale','Vaurien','Vautour','Vicelard','Vieux Chnoque','Vieux Con','Vieux Fossile',
    'Vieux Tableau','Vieux Tromblon','Vilain Comme Une Couvée De Singe','Vioque','Voleur','Voyou',
    'Wisigoth',
    'Yéti',
    'Zigomar','Zigoto','Zonard','Zouave','Zoulou','Zozo','Zéro'
]

gros_mots_féminin = [
    'Andouille',
    'Banane','Betterave','Bougre De Conne','Boule De Pus','Bourique','Bourse molle','Boursouflure','Brêle','Brosse à Chiottes','Burne','Bécasse',
    'Cagole','Canaille','Catin','Cervelle D’huitre','Charogne','Cloche','Connasse','Conne','Cornegidouille',
    'Couille De Tétard','Couille Molle','Crapule','Crevure','Crotte De Moineau',
    'Enflure','Erreur De La Nature',
    'Face De Cul','Face De Pet','Face De Rat','Fesse D’huitre','Fesse De Moule','Fesses Molles','Fiente de moineau',
    'Flaque De Pus','Foldingue','Frapadingue','Fripouille',
    'Gangrène','Godiche','Gourdasse','Grenouille','Grognasse','Grosse Merde Puante','Grosse Truie Violette','Grue','Gueule De Fion','Gueule De Raie',
    'Infâme Raie De Cul','Ironie De La Création',
    'Larve','Loutre Analphabète',
    'Maquerelle','Merdasse','Merde','Merde Molle','Morue','Mortecouille',
    'Nounouille',
    'Ordure Purulente','Outre à Pisse',
    'Patate','Peau De Bite','Peau De Vache','Petite Merde','Petzouille','Pimbêche','Pine D’ours','Pine D’huitre','Pintade',
    'Pipistrelle Puante','Piqueniquedouille','Pisseuse','Pissure','Pompe A Merde','Pouffe','Pouffiasse','Pourriture','Punaise',
    'Pute Au Rabais','Pute Borgne','Putréfaction','Pétasse','Pétassoïde Conassiforme',
    'Quiche',
    'Raclure De Bidet','Raclure De Chiotte','Radasse','Roulure',
    'Saleté','Salope','Saloperie','Serpillière à Foutre','Sombre Conne','Souillon','Sous Merde','Suintance',
    'Tanche','Tartignole','Tasse à Foutre','Trainée','Triple Buse','Tronche De Cake',
    'Tête D’ampoule','Tête De Bite','Tête De Chibre','Tête De Con','Tête De Noeud','Tête à Claques',
    'Vermine','Vieille Baderne','Vieille Poule','Vieille Taupe','Vipère Lubrique','Vérole'
]

gros_mots_neutre = [
    'Branque','Bégueule','Bête',
    'Casse-pieds','Chafouin,','Cinglé','Con',
    'Fini à L’urine','Fourbe',
    'Gras Du Bide',
    'Has-been',
    'Outrecuidant',
    'Veule','Vilain'
]

compléments = [
    'ambulant','anémique',
    'baveux','bouseux','bourin',
    'cinglé','cocu','crasseux',
    'débile','décérébré','dégueulasse','dégénéré','dépravé','desséché',
    'écervelé','emplâtré','empoté',
    'foireux','frigide','frustré',
    'graveleux',
    'illettré','imbibé','immonde','impuissant',
    'lobotomisé',
    'malaimé','mal baisé','méchant','minable','misérable','miteux','moche','mononeuronal','mou',
    'naze','niaiseux',
    'pouilleux','pourri','puant',
    'répugnant',
    'sans le sou','sinoque','sodomite','syphonné',
    'taré'
]

insultes_bizarres = [
    'T’es con comme du plastique.',
    'T’es comme une pizza, sauf qu’elle on peut l’avoir sans champignons.',
    'Tête d’endive au jambon.',
    'Ton père c’est un fils de poutre et ta sœur une chaise pliante.',
    'Tu es con comme un pied d’chaise.',
    'Espèce de pelle à tarte.',
    'Tes oreilles on dirait des pantoufles.',
    'Je vais casser les lavabos de toute ta famille.',
    'Ta mère elle a pas Netflix.',
    'Couillon d’la lune.',
    'Fils de mouette.',
    'Tu ferais gerber un renard.',
    'Gobe-moi les ovaires et étouffe-toi avec.',
    'Tête de chips.',
    'Ton nez on dirait un escalator de 2 kilomètres.',
    "Dieu a gâché un bon trou du cul quand il a mis des dents dans ta bouche.",
    "En voulant récupérer un papier, j'ai retrouvé ta dignité dans la poubelle. J'imagine que tu n'en auras pas besoin.",
    "Peut-être que tu serais moins con si tes parents étaient cousins au deuxième degré et pas au premier.",
    "T'es aussi intéressant que l'autobiographie de Patrick Bosso.",
    "Regarde où tu mets les pieds, tu risques de marcher sur la merde que t'es en train de raconter.",
    "Je vais t'arracher la tête et chier dans ton cou.",
    "Remarque très intéressante : ce que je te propose, c'est que j'en fasse une petite boule de papier, que je la jette par la fenêtre, puis je descende les escaliers pour la récupérer afin de la remonter et la mettre cordialement dans ton cul.",
    "Tu es dispo le soir ? J'aimerais que tu viennes me parler, ça me permettrait de m'endormir plus vite.",
    "J'ai demandé à être muté en Syrie pour ne plus voir ta gueule.",
    "Une question : ça existe, le délit d'outrage à connard ?",
    "Tu as vraiment une peau magnifique. Tu donnerais tout ton potentiel en descente de lit.",
    "J'aimerais tellement partir au bout du monde avec toi... On prendrait l'avion, on serait seuls, libres ; on s'ébattrait. Puis je t'abandonnerais et je rentrerais en France ni vu ni connu.",
    "T'es tellement con que si les cerveaux étaient taxés, t'aurais un énorme avoir fiscal!",
    "T'es tellement con que tu vas à la caisse d'épargne pour ouvrir des noisettes!",
    "T'es tellement teubé que t'es capable de prendre une télécommande quand tu vas au cinéma!",
    "T'es tellement con que tu dois échouer même à tes examens d'urine!",
    "J'ai l'impression que t'as oublié de payer la taxe sur le cerveau toi!",
    "Tu es aussi sexy qu'un tracteur à pelouse mon pauvre",
    "Ton visage restera gravé dans mon cœur comme un cancer dans un poumon de vieux fumeur!",
    "Tu sais, l'homme descend du singe, le singe descend de l'arbre, mais toi mon pauvre t'as dû rater une branche!",
    "Si la connerie était cotée en bourse, tu serais incarcéré pour délit d'initié!",
    "Jolie ta culture de champignons sur ta gueule. Pas besoin d'en acheter chez Panzani pour mettre sur la pizza!"
]


def copie_liste_d_insultes(liste):
    résultat = []
    for mot in liste:
        résultat.append(mot.lower())
    return résultat


def conjuguer_un_complément(complément, genre):
    if genre != 1:
        return complément

    if complément[-1] == 'e':
        return complément
    elif complément[-1] == 'é':
        return complément + 'e'
    elif complément.find(' ') > 0:
        return complément
    elif complément == 'mou':
        return 'molle'
    if complément[-1] in ('i', 'l', 'u'):
        return complément + 'e'
    elif complément[-2:] == 'in':
        return complément + 'ne'
    elif complément[-3:] == 'eux':
        return complément[:-3] + 'euse'
    elif complément[-3:] == 'ant':
        return complément + 'e'
    else:
        return complément


def un_gros_mot():
    genre = random.randint(0,2)
    liste_de_mots = [gros_mots_masculin, gros_mots_féminin, gros_mots_neutre][genre]
    insulte = liste_de_mots[random.randint(0, len(liste_de_mots)-1)]
    if genre == 0:
        return "un " + insulte
    elif genre == 1:
        return "une " + insulte
    else:
        return insulte


def une_insulte_au_hasard():
    return insultes_bizarres[random.randint(0, len(insultes_bizarres)-1)]


def générer_une_insulte(attaque, gmm, gmf, gmn, pb):
    if len(gmm) + len(gmf) + len(gmn) + len(pb) == 0:
        return None, [], [], [], []

    genre = (0, 0, 0, 0, 0, 1, 1, 1, 2, 3)[random.randint(0,9)]
    liste_choisie = (gmm, gmf, gmn, pb)[genre]
    taille_liste = len(liste_choisie)
    while taille_liste == 0:
        genre = genre + 1 if genre < 3 else 0
        liste_choisie = (gmm, gmf, gmn, pb)[genre]
        taille_liste = len(liste_choisie)
    
    mot_choisi = random.randint(0, taille_liste-1)
    if genre == 0:
        insulte = liste_choisie[mot_choisi]
        gmm.remove(insulte)
        if random.randint(0,1) == 1:
            complément = compléments[random.randint(0, len(compléments)-1)]
            insulte += " " + conjuguer_un_complément(complément, genre)
        réponse = "Tu es un " + insulte
    elif genre == 1:
        insulte = liste_choisie[mot_choisi]
        gmf.remove(insulte)
        if random.randint(0,1) == 1:
            complément = compléments[random.randint(0, len(compléments)-1)]
            insulte += " " + conjuguer_un_complément(complément, genre)
        réponse = "Tu es une " + insulte
    elif genre == 2:
        insulte = liste_choisie[mot_choisi]
        gmn.remove(insulte)
        réponse = "Tu es " + insulte
    else:
        insulte = liste_choisie[mot_choisi]
        pb.remove(insulte)
        réponse = insulte

    return réponse, gmm, gmf, gmn, pb


def dire_à_haute_voix(phrase, bufferisable = False):
    if audio:
        fichier = os.path.join("buffer_audio", str(hash(phrase)) + ".mp3")
        
        if not os.path.isdir('buffer_audio'):
            os.mkdir("buffer_audio")

        if not bufferisable or not os.path.isfile(fichier):
            myobj = gTTS(text=phrase, lang="fr", slow=False)
            myobj.save(fichier)
        
        print(phrase)
        os.system("ffplay -nodisp -autoexit " + fichier + " >/dev/null 2>&1")
        
        if not bufferisable:
            os.remove(fichier)
    else:
        print(phrase)


def écouter_une_phrase(recognizer, microphone):
    global audio

    if audio == False:
        return None

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("J'écoute...")
        audio = recognizer.listen(source, phrase_time_limit=5)
        print("... c'est entendu!")

    try:
        text = recognizer.recognize_google(audio, language='fr-FR') 
        print("Vous avez dit: " + text)
        return text

    #error occurs when google could not understand what was said 
    except sr.UnknownValueError:
        dire_à_haute_voix("Je n'ai rien compris, il faut savoir articuler") 
        return None
      
    except sr.RequestError as e: 
        dire_à_haute_voix("Je ne comprends rien, on va dire que vous êtes muet")
        audio = False
        return None 


def cibler_une_insulte(phrase):
    phrase = phrase.lower()
    mots_de_la_phrase = word_tokenize(phrase)
 
    insulte = []
    for mot in mots_de_la_phrase:
        if mot not in stop_words:
            insulte.append(mot)
 
    print(insulte)
    return insulte


if __name__ == "__main__":

    # insulte = cibler_une_insulte("Ta mère elle n'a pas Netflix, tu es un gros naze.")

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    random.seed()

    dire_à_haute_voix("Salut, comment t'appelles-tu ?", True)
    joueur = écouter_une_phrase(recognizer, microphone)
    if joueur is None:
        joueur = "L'inconnu"
        dire_à_haute_voix("Puisque tu ne me dis rien, je vais t'appeler l'inconnu.", True)
    else:
        dire_à_haute_voix("Bonjour " + joueur + ", tu es " + un_gros_mot())

    gmm = copie_liste_d_insultes(gros_mots_masculin)
    gmf = copie_liste_d_insultes(gros_mots_féminin)
    gmn = copie_liste_d_insultes(gros_mots_neutre)
    pb = insultes_bizarres.copy()

    déjà_entendu = []
    points = 10

    i = 50
    ras_le_bol = 10
    while i > 0 and points > 0:

        phrase = écouter_une_phrase(recognizer, microphone)
        if phrase is not None:

            insulte = cibler_une_insulte(phrase)

            # Tranformer l'insulte en string
            insulte = ' '.join(insulte)

            if 'stop' in insulte:
                dire_à_haute_voix("Tu as perdu, tu n'as pas su me faire rire. Salut!")
                break

            print("Ton insule : " + str(insulte))
            print(déjà_entendu)

            # Vérifier que l'insulte n'a pas déjà été entendue
            if insulte in déjà_entendu:
                dire_à_haute_voix("Tu me l'as déjà dit, tu perds un point", True)
                points = points - 1
                dire_à_haute_voix("Il te reste " + str(points) + " points.")
            else:        
                # enregistrer l'insulte dans le tableau des insultes déjà entendues
                déjà_entendu.append(insulte)

            phrase, gmm, gmf, gmn, pb = générer_une_insulte(insulte, gmm, gmf, gmn, pb)
            if phrase is None:
                break
        
            dire_à_haute_voix(phrase)
            i = i-1

        else: 
            ras_le_bol -= 1
            if ras_le_bol == 0:
                dire_à_haute_voix("Il semble que tu sois parti pleurer chez ta mère. Salut!")
                break

    print("Terminé :-)")


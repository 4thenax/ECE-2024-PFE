# Importation de la bibliothèque Streamlit
import streamlit as st
from music21 import stream, note, converter, metadata
import subprocess
import os
import shlex

import cv2
import numpy as np
from sklearn.cluster import KMeans
from midiutil import MIDIFile
import matplotlib.pyplot as plt

from pydub import AudioSegment
import base64

import mido
from mido import MidiFile, MidiTrack, Message

import pygame
import random
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Détection objets:
from ultralytics import YOLO
from PIL import Image


# ---------------------------------- Consignes pour run le code :
# Dans terminal windows (ou invité de commande), tapez:

# powershell
# cd D:\-PFE\ECE-2024-PFE      (nom du dossier où est enregistré le fichier)
# streamlit run application.py


#ls : pour voir le contenu du dossier
# cd : pour accéder à un chemin
# powershell : version + puissante du terminal


# ------------------------------------------------------------------------------
# AUDIO + IA


categories = {
    "Aircraft": "plane, airplane, airport",
    "Animals": "cat, farm, donkey, pig, cow, duck, goose, pigeon, horse, ox, ram, buffalo, sheep, ice_bear, dog, bird",
    "Applause":"applause, show, performance, scene, cabaret",
    "Atmosphere":"atmosphere",
    "Bells":"bell, church, altar",
    "Birds":"bird, goose",
    "Clocks":"clock, ticker, time",
    "Crowds":"crowd, people",
    "Daily Life" : "chill, coffee_shop, coffee, library, book, restaurant, tray",
    "Destruction": "destruction",
    "Electronics":"machine, computer, electronic",
    "Events":"firework, festival, party, concert",
    "Fire":"fire, wood, campfire, tents, camping",
    "Footsteps":"footsetps, run, walk",
    "Machines":"machine",
    "Medical":"hospital, sick, invalid, ill, unhealthy",
    "Military":"soldier, war, military, weapon, gun, battle, fog",
    "Nature": "tree, bench, park_bench, sun, water, sea, sunset, seaside, valley, forest, fountain, lakeside, sand, cliff, ice floe, palm, cascade, flower",
    "Sports":"tennis, basketball, ball, football, swimming-pool, swimming, horse, horse racing, boat",
    "Toys":"toy, children toy, puzzle",
    "Transport":"train, station train, cars, bus, taxi", 
}

categories_instruments = {
    "Aircraft": (96, 103),  # SFX Sci-fi à SFX Atmosphere
    "Animals": (113, 123),  # Tinkle Bell à Bird Tweet
    "Applause": (126, 127),  # Applause
    "Atmosphere": (88, 94),  # New Age Syn Pad à Halo Syn Pad
    "Bells": (13, 14),  # Xylophone, Tubular Bells
    "Birds": (72, 78),  # Piccolo à Ocarina
    "Clocks": (0, 7),  # Acoustic Grand Piano à Clavinet
    "Crowds": (48, 62),  # String Ensemble 1 à Syn Brass 2
    "Daily Life": (24, 31),  # Guitar
    "Default": (40, 47), # Violin à Timpani
    "Destruction": (116, 118),  # Melodic Tom à Syn Drum
    "Electronics": (80, 127),  # Syn Square Wave à Gun Shot
    "Events": (48, 62),  # String Ensemble 1 à Syn Brass 2
    "Fire": (97, 99),  # SFX Soundtrack à SFX Brightness
    "Footsteps": (115, 115),  # Woodblock
    "Machines": (0, 118),  # Acoustic Grand Piano à Syn Drum
    "Medical": (88, 94),  # New Age Syn Pad à Halo Syn Pad
    "Military": (56, 61),  # Trumpet à Brass Section
    "Nature": (72, 76),  # Piccolo à Bottle Blow
    "Sports": (56, 62),  # Trumpet à Syn Brass 2
    "Toys": (112, 118),  # Tinkle Bell à Syn Drum
    "Transport": (97, 127)  # SFX Soundtrack à Gun Shot
}


categorie_emotion = {
    "bicycle" : "Querelleux et criard", "car" : "Querelleux et criard", "motorcycle" : "Querelleux et criard", "airplane" : "Magnifique et joyeux",
    "bus" : "Querelleux et criard", "train" : "Querelleux et criard", "truck" : "Querelleux et criard", "boat" : "Magnifique et joyeux", "bench" : "Solitaire et melancolique",
    "bird" : "Joyeux et champetre", "cat" : "Joyeux et champetre", "dog" : "Joyeux et champetre", "horse" : "Joyeux et champetre",
    "sheep" : "Joyeux et champetre", "cow" : "Joyeux et champetre", "elephant" : "Joyeux et champetre", "bear" : "Joyeux et champetre", "zebra" : "Joyeux et champetre",
    "giraffe" : "Joyeux et champetre", "umbrella" : "Obscur et plaintif", "handbag" : "Grave et devot", "suitcase" : "Magnifique et joyeux", "tie" : "Furieux et emporte",
    "frisbee" : "Joyeux et tres guerrier", "snowboard" : "Joyeux et tres guerrier", "sports ball" : "Joyeux et tres guerrier", "kyte" : "Magnifique et joyeux",
    "baseball bat" : "Joyeux et tres guerrier", "baseball glove" : "Joyeux et tres guerrier", "skateboard" : "Joyeux et tres guerrier", "surfboard" : "Joyeux et tres guerrier",
    "tennis racket" : "Joyeux et tres guerrier", "bottle" : "Grave et devot", "wine glass" : "Furieux et emporte", "cup" : "Grave et devot", "spoon" : "Grave et devot",
    "knife" : "ton du fossoyeur", "fork" : "Grave et devot", "bowl" : "Grave et devot", "banana" : "Joyeux et champetre", "apple" : "Joyeux et champetre",
    "sandwich" : "Joyeux et champetre", "orange" : "Joyeux et champetre", "broccoli" : "Joyeux et champetre", "carrot" : "Joyeux et champetre", "hot dog" : "Magnifique et joyeux",
    "pizza" : "Magnifique et joyeux", "donut" : "Magnifique et joyeux", "cake" : "Magnifique et joyeux", "chair" : "Grave et devot", "couch" : "Grave et devot",
    "potted plant" : "Joyeux et champetre", "bed" : "Tendre et plaintif", "dining table" : "Grave et devot", "toilet" : "Grave et devot", "tv" : "Grave et devot",
    "laptop" : "Magnifique et joyeux", "mouse" : "Magnifique et joyeux", "remote" : "Magnifique et joyeux", "keyboard" : "Magnifique et joyeux",
    "cell phone" : "Magnifique et joyeux", "microwave" : "Grave et devot", "oven" : "Grave et devot", "hair drier" : "Grave et devot", "toothbrush" : "Grave et devot",
    "teddy bear" : "Serieux et magnifique", "vase" : "Doucement joyeux", "clock" : "Solitaire et melancolique", "book" : "Serieux et magnifique", "scissors" : "ton du fossoyeur",
    "person" : "Serieux et magnifique", "skies" : "Joyeux et tres guerrier"
}

# Associations des émotions aux tonalités

# do (C), ré (D), mi(E), fa (F), sol (G), la (A), si (B).
categorie_tonalité = {
    "Querelleux et criard" : "E",
    "Magnifique et joyeux" : "Bb", 
    "Solitaire et melancolique" : "B",
    "Joyeux et champetre" : "A",
    "Obscur et plaintif" : "F",
    "Furieux et emporte" : "F",
    "Joyeux et tres guerrier" : "D",
    "Grave et devot" : "D",
    "Tendre et plaintif" : "A",
    "Serieux et magnifique" : "G",
    "Doucement joyeux" : "G",
    "ton du fossoyeur" : "Ab"
}

categorie_tonalité2 = {
    "Querelleux et criard" : "Majeur",
    "Magnifique et joyeux" : "Majeur",
    "Solitaire et melancolique" : "mineur",
    "Joyeux et champetre" : "Majeur",
    "Obscur et plaintif" : "mineur",
    "Furieux et emporte" : "Majeur",
    "Joyeux et tres guerrier" : "Majeur",
    "Grave et devot" : "mineur",
    "Tendre et plaintif" : "mineur",
    "Serieux et magnifique" : "mineur",
    "Doucement joyeux" : "Majeur",
    "ton du fossoyeur" : "Majeur"
}

nom_instru = {
    0: "Acoustic Grand Piano",
    1: "Bright Acoustic Piano",
    2: "Electric Grand Piano",
    3: "Honky Tonk Piano",
    4: "Electric Piano 1",
    5: "Electric Piano 2",
    6: "Harpsichord",
    7: "Clavinet",
    8: "Celesta",
    9: "Glockenspiel",
    10: "Music Box",
    11: "Vibraphone",
    12: "Marimba",
    13: "Xylophone",
    14: "Tubular Bells",
    15: "Dulcimer",
    16: "Drawbar Organ",
    17: "Percussive Organ",
    18: "Rock Organ",
    19: "Church Organ",
    20: "Reed Organ",
    21: "Accordion",
    22: "Harmonica",
    23: "Tango Accordion",
    24: "Nylon Acoustic Guitar",
    25: "Steel Acoustic Guitar",
    26: "Jazz Electric Guitar",
    27: "Clean Electric Guitar",
    28: "Muted Electric Guitar",
    29: "Overdrive Guitar",
    30: "Distorted Guitar",
    31: "Guitar Harmonics",
    32: "Acoustic Bass",
    33: "Electric Fingered Bass",
    34: "Electric Picked Bass",
    35: "Fretless Bass",
    36: "Slap Bass 1",
    37: "Slap Bass 2",
    38: "Syn Bass 1",
    39: "Syn Bass 2",
    40: "Violin",
    41: "Viola",
    42: "Cello",
    43: "Contrabass",
    44: "Tremolo Strings",
    45: "Pizzicato Strings",
    46: "Orchestral Harp",
    47: "Timpani",
    48: "String Ensemble 1",
    49: "String Ensemble 2 (Slow)",
    50: "Syn Strings 1",
    51: "Syn Strings 2",
    52: "Choir Aahs",
    53: "Voice Oohs",
    54: "Syn Choir",
    55: "Orchestral Hit",
    56: "Trumpet",
    57: "Trombone",
    58: "Tuba",
    59: "Muted Trumpet",
    60: "French Horn",
    61: "Brass Section",
    62: "Syn Brass 1",
    63: "Syn Brass 2",
    64: "Soprano Sax",
    65: "Alto Sax",
    66: "Tenor Sax",
    67: "Baritone Sax",
    68: "Oboe",
    69: "English Horn",
    70: "Bassoon",
    71: "Clarinet",
    72: "Piccolo",
    73: "Flute",
    74: "Recorder",
    75: "Pan Flute",
    76: "Bottle Blow",
    77: "Shakuhachi",
    78: "Whistle",
    79: "Ocarina",
    80: "Syn Square Wave",
    81: "Syn Sawtooth Wave",
    82: "Syn Calliope",
    83: "Syn Chiff",
    84: "Syn Charang",
    85: "Syn Voice",
    86: "Syn Fifths Sawtooth Wave",
    87: "Syn Brass & Lead",
    88: "New Age Syn Pad",
    89: "Warm Syn Pad",
    90: "Polysynth Syn Pad",
    91: "Choir Syn Pad",
    92: "Bowed Syn Pad",
    93: "Metal Syn Pad",
    94: "Halo Syn Pad",
    95: "Sweep Syn Pad",
    96: "SFX Rain",
    97: "SFX Soundtrack",
    98: "SFX Crystal",
    99: "SFX Atmosphere",
    100: "SFX Brightness",
    101: "SFX Goblins",
    102: "SFX Echoes",
    103: "SFX Sci-fi",
    104: "Sitar",
    105: "Banjo",
    106: "Shamisen",
    107: "Koto",
    108: "Kalimba",
    109: "Bag Pipe",
    110: "Fiddle",
    111: "Shanai",
    112: "Tinkle Bell",
    113: "Agogo",
    114: "Steel Drums",
    115: "Woodblock",
    116: "Taiko Drum",
    117: "Melodic Tom",
    118: "Syn Drum",
    119: "Reverse Cymbal",
    120: "Guitar Fret Noise",
    121: "Breath Noise",
    122: "Seashore",
    123: "Bird Tweet",
    124: "Telephone Ring",
    125: "Helicopter",
    126: "Applause",
    127: "Gun Shot"
}


st.image("./enfants_musique.jpg")

st.title("Générez votre mélodie à partir d'une image")

# Récupération des variables
nom_image = st.text_input("Saisissez le nom de votre image (avec l'extension) : ")

# Vérifier si l'image a été chargée correctement
script_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_directory, nom_image)

if nom_image == '':
    print('')
elif os.path.exists(image_path):
    st.success(f"L'image {nom_image} a été chargée avec succès.")
else :
    st.error(f"L'image avec le nom {nom_image} n'existe pas dans le répertoire actuel.")



## tonalité et mode déterminé à partir des objets présents sur l'image
# Modèle de detection (YOLOv8)

# Définition de la fonction dans le même fichier
def get_variable() :
    objet = label
    emotion = emotion_associe
    tonalite = tonalite_associe
    tonalite2 = tonalite_associe2
    return objet, emotion, tonalite, tonalite2
    
if image_path.strip():
    
    # Vérifier si le chemin de l'image est valide
    if os.path.exists(image_path):
        model = YOLO("yolov8m.pt")

        results = model.predict(image_path)

        result = results[0]
        print(len(result.boxes))

        for box in result.boxes :
            label = result.names[box.cls[0].item()]
            cords = [round(x) for x in box.xyxy[0].tolist()]
            prob = round(box.conf[0].item(), 2)
            class_id = box.cls[0].item()
            print("Object: ", label)
            print("coordonnées:", cords)
            print ("probabilité:", prob)
            print("classe de l'objet:", class_id)
            
            # Trouver la catégorie associée à l'émotion
            emotion_associe = categorie_emotion.get(label, "Inconnu")
            st.write("Emotion:", emotion_associe)
            tonalite_associe = categorie_tonalité.get(emotion_associe, "Inconnu")
            st.write("Tonalité:", tonalite_associe)
            tonalite_associe2 = categorie_tonalité2.get(emotion_associe, "Inconnue")
            st.write("Mode:", tonalite_associe2)
            print()

            # Appel de la fonction
            objet, emotion, tonalite, mode = get_variable()

    else:
        print("Le chemin de l'image spécifié n'est pas valide.")
else:
    print("Le chemin de l'image n'a pas été spécifié.")


nom_audio = st.text_input("Saisissez le nom que vous voulez donner à votre fichier audio (sans l'extension) : ")


############### IA POUR TROUVER THEME IMAGE ET L'ASSOCIER A UNE CATEGORIE


def trouver_categorie(labels, categories):
    vect = TfidfVectorizer()
    descriptions = list(categories.values())
    vect.fit(descriptions)
    cat_vectors = vect.transform(descriptions)

    label_vector = vect.transform([" ".join(labels)])
    sim_scores = cosine_similarity(label_vector, cat_vectors)

    # Trouver la catégorie avec le score de similarité le plus élevé
    categorie_index = np.argmax(sim_scores)

    # Vérifier si le score est inférieur à un certain seuil (par exemple, 0.5)
    seuil_similarity = 0.3
    if sim_scores[0, categorie_index] < seuil_similarity:
        # Aucun mot n'est présent, retourner la catégorie par défaut
        return "Default"
    
    else:
        # Au moins un mot est présent, retourner la catégorie correspondante
        categorie = list(categories.keys())[categorie_index]
        return categorie


# initialisation de catégorie
categorie = None

if image_path and os.path.isfile(image_path):
        model = MobileNetV2(weights='imagenet')

        # Charger l'image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prétraitement pour le modèle
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Prédiction
        prediction = model.predict(image)
        labels = decode_predictions(prediction)
        label1, label2, label3 = None, None, None

        if len(labels[0]) > 0:
            label1 = labels[0][0][1]

        if len(labels[0]) > 1:
            label2 = labels[0][1][1]

        if len(labels[0]) > 2:
            label3 = labels[0][2][1]

        # Afficher les labels
        print("Label 1:", label1)
        print("Label 2:", label2)
        print("Label 3:", label3)

        labels_extraits = [label1, label2, label3]
        st.write("Labels extraits :", labels_extraits)
        categorie = trouver_categorie(labels_extraits, categories)
        st.write("Catégorie correspondante :", categorie)
        
else:
    st.warning("Le chemin de l'image n'a pas été spécifié ou n'est pas un fichier existant.")


# Saisie manuelle de la catégorie :
# categorie = st.text_input("Saisir la catégorie de l'image, définie par le modèle IA entraîné")

#Fonction qui ressort la liste des instruments correspondants à telle catégorie
def instruments_possibles(categorie, categories_instruments):
    if categorie in categories_instruments:
        debut, fin = categories_instruments[categorie]
        liste_instruments = [int(i) for i in range(debut, fin + 1)]
        return liste_instruments
    else:
        return []

def get_instru(numero):
    # Vérifier si le numéro est dans la plage valide
    if 0 <= numero <= 127:
        return nom_instru.get(numero, "Instrument inconnu")
    else:
        return "Numéro d'instrument invalide"


# Liste d'instruments possibles pour cette catégorie
liste_instruments_correspondants = instruments_possibles(categorie, categories_instruments)
st.write("Liste d'instruments possibles :")

# Affiche le nom de l'instrument correspondant selon tableau MIDI
for numero in liste_instruments_correspondants:
    nom_instrument = get_instru(numero)
    if "invalide" in nom_instrument:
        st.write(f"Le numéro d'instrument {numero} est invalide.")
    else:
        st.write(f"{numero} = {nom_instrument}")


### Saisie manuelle de la tonalité :

# mode = st.text_input("Entrez le mode du morceau (Majeur ou mineur) : ")
# Vérification du mode saisi
# if mode == '':
#     print('')
# elif mode not in ['Majeur', 'mineur']:
#     st.warning("Le mode doit être 'Majeur' ou 'mineur'. Veuillez saisir une valeur valide.")

# tonalite = st.text_input("Entrez la tonalité du morceau : ")
# st.text("C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B")
# allowed_tonalities = {'C': 0, 'C#': 1, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 5, 'E': 6, 'F': 7, 'F#': 8, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 12, 'A': 13, 'A#': 14, 'Bb': 15, 'B': 16}
# if tonalite == '':
#     print('')
# elif tonalite not in allowed_tonalities:
#     st.warning("Tonalité invalide. Veuillez saisir une tonalité parmi les valeurs autorisées.")


# Saisie de la séquence de rythme

sequence_input = st.text_input("Saisissez la séquence de rythme voulue (sans virgules entre les chiffres) : ")
st.write("Par exemple : 1 2 0.5 0.5 0.5 1 1 0.5 0.5 1")

sequence = [float(duration) for duration in sequence_input.split(" ") if duration.strip() != '']


# Fonction pour extraire les couleurs quantifiées
def extract_quantized_colors(image, num_colors=50):
    # Redimensionner l'image pour réduire le temps de traitement
    resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    # Remodeler l'image pour être une liste de pixels
    pixels = resized_image.reshape((-1, 3))

    # Convertir de BGR à RGB pour KMeans
    pixels = np.float32(pixels[:, ::-1])

    # Utiliser KMeans pour quantifier les couleurs
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)

    # Les centres de cluster sont les couleurs quantifiées
    quantized_colors = kmeans.cluster_centers_

    # Convertir les couleurs en entiers
    quantized_colors = np.uint8(quantized_colors)

    # Convertir les couleurs en RGB standard pour l'affichage
    quantized_colors = quantized_colors[:, ::-1]

    return quantized_colors


# Fonction pour mapper les couleurs aux notes
def color_to_pitch(color):
    # Calculer la note de base à partir de la couleur
    base_pitch = int(sum(color)) % 128      # % = modulo le nb de notes dans la table MIDI
    adapted_pitch = base_pitch % 36 + 48
    
    return adapted_pitch


# Fonction pour ajuster les notes en fonction de la tonalité choisie
def adjust_notes_to_key(notes, tonalite):
    # Définir la correspondance des notes dans la tonalité choisie
    key_mapping = {'C': 0, 'C#': 1, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 5, 'E': 6, 'F': 7, 'F#': 8, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 12, 'A': 13, 'A#': 14, 'Bb': 15, 'B': 16}

    # Trouver la transposition nécessaire pour correspondre à la tonalité choisie
    transposition = key_mapping[tonalite] - key_mapping['C']

    # Ajuster chaque note en fonction de la transposition
    adjusted_notes = [(note + transposition) % 128 for note in notes]

    return adjusted_notes


def create_melody(notes, mode):
    scale = []

    if mode == "Majeur":
        scale = [0, 2, 4, 5, 7, 9, 11]
    elif mode == "mineur":
        scale = [0, 2, 3, 5, 7, 8, 10]
    else:
        print("erreur dans la saisie du mode")
 
    melody = []
    prev_note = notes[0]

    for note in notes:
        # Ajouter une variation autour de la note précédente
        variation = np.random.randint(-1, 2)  # Variation de -1, 0, ou 1
        new_note = max(0, min(127, prev_note + variation))

        # Utiliser la gamme majeure pour créer des motifs mélodiques
        melodic_offset = scale[new_note % len(scale)]

        melody.append(new_note + melodic_offset)
        prev_note = new_note

    return melody


def convert_midi_to_audio(midi_file_path):
    audio = AudioSegment.from_file(midi_file_path, format="midi")
    audio_path = f"./{nom_audio}.wav"
    audio.export(audio_path, format="wav")
    return audio_path


def change_instrument(input_file, output_file, new_instrument1, new_instrument2):
    try:
        # Charger le fichier MIDI d'entrée
        midi_file = MidiFile(input_file)

        # Créer un nouveau fichier MIDI
        new_midi_file = MidiFile()

        # Parcourir toutes les pistes du fichier MIDI d'entrée
        for i, track in enumerate(midi_file.tracks):
            new_track1 = MidiTrack()
            new_midi_file.tracks.append(new_track1)

            new_track2 = MidiTrack()
            new_midi_file.tracks.append(new_track2)
            # Indicateur pour savoir si le message "Program Change" a été ajouté à cette piste
            program_change_added1 = False
            program_change_added2 = False

            # Parcourir tous les messages sur la piste
            for msg in track:
                # Vérifier si le message est du type "Program Change" (changement d'instrument)
                if msg.type == 'program_change':
                    # Mettre à jour l'instrument pour chaque piste
                    if not program_change_added1:
                        new_track1.append(Message('program_change', program=new_instrument1, time=msg.time))
                        program_change_added1 = True

                    if not program_change_added2:
                        new_track2.append(Message('program_change', program=new_instrument2, time=msg.time))
                        program_change_added2 = True
                else:
                    # Copier les autres messages tels quels
                    new_track1.append(msg)
                    new_track2.append(msg)

            # Si aucun message "Program Change" n'a été trouvé sur la piste, l'ajouter
            if not program_change_added1:
                new_track1.append(Message('program_change', program=new_instrument1, time=0))

            if not program_change_added2:
                new_track2.append(Message('program_change', program=new_instrument2, time=0))

        # Sauvegarder le nouveau fichier MIDI
        new_midi_file.save(output_file)

        print(f"L'instrument a été changé avec succès dans le fichier {output_file}")

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")




if st.button("Générer l'audio"):

    # Charger l'image
    image = cv2.imread(image_path)

    # Vérifier si l'image a été chargée correctement
    if image is None:
        st.warning("Erreur lors du chargement de l'image. Vérifiez le chemin du fichier.")
    else:

        # Vérifier si tonalite et mode sont définis, sinon assigner des valeurs par défaut
        if 'tonalite' not in locals() or 'mode' not in locals():
            st.write("Aucune correspondance de tonalité n'a été trouvée pour cette image. Des valeurs par défaut ont été assignées.")
            tonalite = "C"
            mode = "Majeur"

        # Extraire les couleurs quantifiées
        colors = extract_quantized_colors(image, num_colors=50)

        # Ajouter des notes basées sur les couleurs quantifiées
        channel = 0
        volume = 100

        # Créer le fichier MIDI
        midi = MIDIFile(1)
        track = 0
        time = 0
        midi.addTrackName(track, time, "Track")
        midi.addTempo(track, time, 120)

        # Spécifier l'instrument avant d'ajouter des notes à la piste
        # instrument = random.choice(liste_instruments_correspondants)
        # st.write(instrument)
        # midi.addProgramChange(track, channel, time, instrument)

        # Ajouter des notes basées sur les couleurs quantifiées
        notes = [color_to_pitch(color) for color in colors]

        adjusted_notes = adjust_notes_to_key(notes, tonalite)
        melody = create_melody(adjusted_notes, mode)

        for i, pitch in enumerate(melody):
            duration = sequence[i % len(sequence)]
            midi.addNote(track, channel, pitch, time, duration, volume)
            time += duration

        instruments_choices = random.sample(liste_instruments_correspondants, 2)

        # Extraire les deux instruments choisis
        instrument1 = instruments_choices[0]
        instrument2 = instruments_choices[1] 
        st.write("Les deux instruments choisis sont:", instrument1, instrument2)
        st.write("Tonalite du morceau:", tonalite, mode)

        # Construction d'un chemin absolu pour le fichier audio
        output_midi_path = os.path.join(script_directory, f"{nom_audio}.mid")

        # Vérifier si le fichier MIDI existe déjà
        if os.path.exists(output_midi_path):
            st.warning('Attention : Un fichier avec le même nom existe déjà. Aucun nouveau fichier n\'a été enregistré.')
        else:
            # Écrire le fichier MIDI
            with open(output_midi_path, 'wb') as outf:
                midi.writeFile(outf)
            
            change_instrument(output_midi_path, output_midi_path, instrument1, instrument2)

            # Vérifier si le fichier MIDI a été enregistré avec succès
            if os.path.exists(output_midi_path):
                st.success(f'Le fichier MIDI a été enregistré avec succès sous : {output_midi_path}')
            else:
                st.error('Erreur lors de l\'enregistrement du fichier MIDI.')



# ------------------------------------------------------------------------------
# PARTITION 

def generate_musescore_from_midi(midi_file_path, output_musicxml_path, titre_partition, nom_compositeur):
    # Charger le fichier MIDI
    midi_stream = converter.parse(midi_file_path)

    # Créer une nouvelle partition MuseScore
    musescore_stream = stream.Score()

    # Ajouter les métadonnées à la partition (titre, compositeur, etc.)
    metadata_obj = metadata.Metadata()
    metadata_obj.title = titre_partition
    metadata_obj.composer = nom_compositeur
    musescore_stream.metadata = metadata_obj

    # Ajouter les parties du fichier MIDI à la partition MuseScore
    for part in midi_stream.parts:
        musescore_stream.append(part)

    # Écrire la partition au format MusicXML
    musescore_stream.write("musicxml", fp=output_musicxml_path)

    # Convertir le fichier MusicXML en MSCZ en utilisant MuseScore en ligne de commande
    command = f"mscore -o {output_musicxml_path}"
    subprocess.run(command, shell=True)



def convert_musicxml_to_pdf(input_musicxml, output_pdf):
    # Construction de la commande

    # Chemin où est installé musescore sur votre ordinateur
    muse_score_path = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"
    
    os.environ["PATH"] += os.pathsep + muse_score_path

   
    command = f"{muse_score_path} -o {output_pdf} {input_musicxml}"

    # Exécution de la commande
    result = subprocess.run(shlex.split(command), stderr=subprocess.PIPE)

    if result.returncode == 0:
        print("Conversion réussie.")
    else:
        print(f"Erreur lors de la conversion. Code de sortie : {result.returncode}")
        print("Erreurs standard (stderr):")
        print(result.stderr.decode("latin-1"))


# Fonction principale pour la page Partition
def partition_page():

    # Titre de la page
    st.title("Créez votre partition : ")

    # Ajout des champs de texte :
    titre_partition = st.text_input("Saisissez le titre de votre partition (sans espace): ")
    nom_compositeur = st.text_input("Saisissez votre nom de compositeur : ")

    # Vérifier si le titre de la partition contient des espaces
    if ' ' in titre_partition:
        st.warning("Le titre de la partition ne doit pas contenir d'espaces. Veuillez saisir un titre sans espaces.")
    elif titre_partition == "":
        st.warning("Le titre de la partition ne peut pas être vide. Veuillez saisir un titre.")

    else:
        
        # Ajout d'un bouton
        if st.button("Générer la partition"):
            # génération de la partition
            midi_file_path = f"./{nom_audio}.mid"
            output_musicxml_path = f"./partitions/{titre_partition}.musicxml"

            if not os.path.exists(output_musicxml_path):
                generate_musescore_from_midi(midi_file_path, output_musicxml_path, titre_partition, nom_compositeur)
                st.write(f"Votre partition '{titre_partition}' a été générée !")
            else:
                st.write(f"Une partition avec le nom '{titre_partition}' existe déjà. Veuillez choisir un autre nom.")


        if st.button("Générer en PDF"):
            # génération en PDF
            output_musicxml_path = f"./partitions/{titre_partition}.musicxml"
            output_pdf = f"./partitions/{titre_partition}.pdf"

            if not os.path.exists(output_pdf):
                convert_musicxml_to_pdf(output_musicxml_path, output_pdf)

                # Vérifier si le fichier PDF a été enregistré avec succès
                if os.path.exists(output_pdf):
                    st.write(f"Votre partition '{titre_partition}' est disponible en PDF.")
                else:
                    st.error("Erreur lors de l'enregistrement du fichier PDF. Veuillez réessayer.")
            else:
                st.write(f"Une partition PDF avec le nom '{titre_partition}' existe déjà. Veuillez choisir un autre nom.")


def menu_page():

    # Titre de la page
    st.sidebar.title("Bienvenue sur Picto Music !")
    # Ajout d'un logo
    logo_path = "./logo2.png"
    st.sidebar.image(logo_path, width=200)

    # Sous-titre
    st.sidebar.header("Découvrez une nouvelle manière de composer de la musique")


# Appel de la fonction principale pour l'exécution de l'application
menu_page()
partition_page()


# ------------------------------------------------------------------------------



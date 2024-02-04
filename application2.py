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
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


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


categories_instruments = {
    "Aircraft": (96, 103),  # SFX Sci-fi à SFX Atmosphere
    "Animals": (113, 123),  # Tinkle Bell à Bird Tweet
    "Applause": (126, 127),  # Applause
    "Atmosphere": (88, 94),  # New Age Syn Pad à Halo Syn Pad
    "Bells": (14, 14),  # Tubular Bells
    "Birds": (72, 78),  # Piccolo à Ocarina
    "Clocks": (0, 20),  # Acoustic Grand Piano à Accordion
    "Crowds": (48, 62),  # String Ensemble 1 à Syn Brass 2
    "Daily Life": (0, 27),  # Acoustic Grand Piano à Clean Electric Guitar
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


nom_audio = st.text_input("Saisissez le nom que vous voulez donner à votre fichier audio (sans l'extension) : ")

categorie = st.text_input("Saisir la catégorie de l'image, définie par le modèle IA entraîné")

#Fonction qui ressort la liste des instruments correspondants à telle catégorie
def instruments_possibles(categorie, categories_instruments):
    if categorie in categories_instruments:
        debut, fin = categories_instruments[categorie]
        liste_instruments = [int(i) for i in range(debut, fin + 1)]
        return liste_instruments
    else:
        return []
    

# Liste d'instruments possibles pour cette catégorie
liste_instruments_correspondants = instruments_possibles(categorie, categories_instruments)
st.write("Instruments possibles :", liste_instruments_correspondants)


mode = st.text_input("Entrez le mode du morceau (majeur ou mineur) : ")
# Vérification du mode saisi
if mode == '':
    print('')
elif mode not in ['Majeur', 'mineur']:
    st.warning("Le mode doit être 'Majeur' ou 'mineur'. Veuillez saisir une valeur valide.")


tonalite = st.text_input("Entrez la tonalité du morceau : ")
st.text("C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B")
allowed_tonalities = {'C': 0, 'C#': 1, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 5, 'E': 6, 'F': 7, 'F#': 8, 'Gb': 9, 'G': 10, 'G#': 11, 'Ab': 12, 'A': 13, 'A#': 14, 'Bb': 15, 'B': 16}
if tonalite == '':
    print('')
elif tonalite not in allowed_tonalities:
    st.warning("Tonalité invalide. Veuillez saisir une tonalité parmi les valeurs autorisées.")


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


#Version de base
def change_instrument2(input_file, output_file, new_instrument1, new_instrument2):
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
            program_change_added = False

            # Parcourir tous les messages sur la piste
            for msg in track:
                # Vérifier si le message est du type "Program Change" (changement d'instrument)
                if msg.type == 'program_change':
                    # Mettre à jour l'instrument
                    new_track1.append(Message('program_change', program=new_instrument1, time=msg.time))
                    new_track2.append(Message('program_change', program=new_instrument2, time=msg.time))

                    program_change_added = True
                else:
                    # Copier les autres messages tels quels
                    new_track1.append(msg)
                    new_track2.append(msg)

            # Si aucun message "Program Change" n'a été trouvé sur la piste, l'ajouter
            if not program_change_added:
                new_track1.append(Message('program_change', program=new_instrument1, time=0))
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
        st.error("Erreur lors du chargement de l'image. Vérifiez le chemin du fichier.")
    else:

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
        instrument = random.choice(liste_instruments_correspondants)
        print(instrument)
        midi.addProgramChange(track, channel, time, instrument)

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
        print("Les deux instruments choisis sont:", instrument1, instrument2)
        

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

    st.image("./automne.jpg")

    # Titre de la page
    st.title("Créez votre partition : ")

    # Ajout des champs de texte :
    titre_partition = st.text_input("Saisissez le titre de votre partition (sans espace !!): ")
    # Vérifier si le titre de la partition contient des espaces
    if ' ' in titre_partition:
        st.warning("Le titre de la partition ne doit pas contenir d'espaces. Veuillez saisir un titre sans espaces.")
    
    nom_compositeur = st.text_input("Saisissez votre nom de compositeur : ")

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



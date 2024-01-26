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


nom_audio = st.text_input("Saisissez le nom de votre fichier audio (sans l'extension) : ")


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
#st.write("vous avez entré la séquence de rythme suivante \n:", sequence)


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


if st.button("Générer l'audio"):

    # Charger l'image
    image = cv2.imread(image_path)

    # Vérifier si l'image a été chargée correctement
    if image is None:
        st.error("Erreur lors du chargement de l'image. Vérifiez le chemin du fichier.")
    else:

        # Extraire les couleurs quantifiées
        colors = extract_quantized_colors(image, num_colors=50)

        # Créer le fichier MIDI
        midi = MIDIFile(1)
        track = 0
        time = 0
        midi.addTrackName(track, time, "Track")
        midi.addTempo(track, time, 120)

        # Ajouter des notes basées sur les couleurs quantifiées
        channel = 0
        volume = 100

        notes = [color_to_pitch(color) for color in colors]
        adjusted_notes = adjust_notes_to_key(notes, tonalite)
        melody = create_melody(adjusted_notes, mode)


        for i, pitch in enumerate(melody):
            duration = sequence[i % len(sequence)]
            midi.addNote(track, channel, pitch, time, duration, volume)
            time += duration

        # Construction d'un chemin absolu pour le fichier audio
        output_midi_path = os.path.join(script_directory, f"{nom_audio}.mid")


        # Vérifier si le fichier MIDI existe déjà
        if os.path.exists(output_midi_path):
            st.warning('Attention : Un fichier avec le même nom existe déjà. Aucun nouveau fichier n\'a été enregistré.')
        else:
            # Écrire le fichier MIDI
            with open(output_midi_path, 'wb') as outf:
                midi.writeFile(outf)
            # Vérifier si le fichier MIDI a été enregistré avec succès
            if os.path.exists(output_midi_path):
                st.success(f'Le fichier MIDI a été enregistré avec succès sous : {output_midi_path}')
            else:
                st.error('Erreur lors de l\'enregistrement du fichier MIDI.')


        # Afficher les couleurs quantifiées
        plt.figure(figsize=(8, 6))
        plt.imshow([colors])
        plt.axis('off')
        plt.show()



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



# Définition de la fonction principale
# Fonction principale pour la page Partition
def partition_page():

    st.image("./automne.jpg")

    # Ajout d'un logo
    # logo_path = "./logo2.png"
    # st.sidebar.image(logo_path, width=200)
    # st.image(logo_path, width=200)

    # Titre de la page
    st.title("Créez votre partition : ")

    # Ajout des champs de texte :
    titre_partition = st.text_input("Saisissez le titre de votre partition : ")
    nom_compositeur = st.text_input("Saisissez votre nom de compositeur : ")
    chemin_audio = f"./{nom_audio}.mid"


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
            st.write(f"Votre partition '{titre_partition}' est disponible en PDF.")
        else:
            st.write(f"Une partition PDF avec le nom '{titre_partition}' existe déjà. Veuillez choisir un autre nom.")

def menu_page():

    # Titre de la page
    st.sidebar.title("Bienvenue sur Picto Music !")
    # st.title("Bienvenue sur Picto Music !")

    # Ajout d'un logo
    logo_path = "./logo2.png"
    st.sidebar.image(logo_path, width=200)
    # st.image(logo_path, width=200)

    # Sous-titre
    st.sidebar.header("Découvrez une nouvelle manière de composer de la musique")
    # st.header("Découvrez une nouvelle manière de composer de la musique")

    # Paragraphe de texte
    st.sidebar.text("Appuyez sur le bouton suivant pour \naccéder à la page de génération \nde partition")
    # st.text("Appuyez sur le bouton suivant pour accéder à la page de génération de partition")


    #if st.sidebar.button("Partition"):
    #    if st.button("Partition"):
    #        partition_page()


# Appel de la fonction principale pour l'exécution de l'application
menu_page()
partition_page()



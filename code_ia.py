import cv2
import numpy as np
from sklearn.cluster import KMeans
from midiutil import MIDIFile
import matplotlib.pyplot as plt
import os

# Saisie du nom de l'image et du fichier audio 
#nom_image = input("Saisissez le nom de votre image (avec l'extension) : ")
nom_image = "image1.jpg"

# Initialisation de la variable du nom de l'audio
#nom_audio = input("Saisissez le nom de votre fichier audio (sans l'extension) : ")
nom_audio = "TESTIMAGE1"

#gamme = input("Entrez le mode du morceau (majeur ou mineur) : ")
mode = "mineur"

#tonalite = int(input("Entrez la tonalité du morceau : "))
#Entrez : 1 pour 'C', 2 pour 'C#/Db', 3 pour 'D', 4 pour 'D#/Eb', 5 pour 'E', 6 pour 'F', 7 pour 'F#/Gb', 8 pour 'G', 9 pour 'G#/Ab', 10 pour 'A', 11 pour 'A#/Bb', 12 pour 'B'],
tonalite = "G"

#sequence = input("Saisissez la séquence de rythme voulue : ").split(" ")
# sequence = [1, 0.25, 0.25, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.25, 0.25, 1] #pour image 2
sequence = [1, 1, 2, 0.5, 0.5, 1, 0.5, 0.5] #pour image 1


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

    # Adapter la note pour qu'elle tombe dans la plage des octaves 0 à 4 (notes 24 à 83 selon table MIDI)
    # adapted_pitch = base_pitch % 60 + 24

    # Adapter la note pour qu'elle tombe dans la plage des octaves 2 à 4 (notes 48 à 83 selon table MIDI)
    adapted_pitch = base_pitch % 36 + 48
    
    return adapted_pitch


# Fonction pour ajuster les notes en fonction de la tonalité choisie
def adjust_notes_to_key(notes, tonalite):
    # Définir la correspondance des notes dans la tonalité choisie
    key_mapping = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

    # Trouver la transposition nécessaire pour correspondre à la tonalité choisie
    transposition = key_mapping[tonalite] - key_mapping['C']

    # Ajuster chaque note en fonction de la transposition
    adjusted_notes = [(note + transposition) % 128 for note in notes]

    return adjusted_notes



def create_melody(notes, mode):
    #ou : create_melody(notes)
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


# Charger l'image
script_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_directory, nom_image)
print(image_path)
# image_path = f"./{nom_image}"
image = cv2.imread(image_path)

# Vérifier si l'image a été chargée correctement
if image is None:
    print("Erreur lors du chargement de l'image. Vérifiez le chemin du fichier.")
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

    # Ajouter des notes à la piste MIDI avec des durées variées
    #duration_sequence = [1, 1, 2, 0.5, 0.5, 1, 0.5, 0.5] # exemple de séquence de rythme
    #duration_sequence = [1, 0.5, 0.5, 1, 0.5, 0.5, 1, 2] # exemple de séquence de rythme

    for i, pitch in enumerate(melody):
        duration = sequence[i % len(sequence)]
        midi.addNote(track, channel, pitch, time, duration, volume)
        time += duration

    # Construct the absolute path for the audio file
    output_midi_path = os.path.join(script_directory, f"{nom_audio}.mid")

    # Écrire le fichier MIDI
    with open(output_midi_path, 'wb') as outf:
        midi.writeFile(outf)

    # Vérifier si le fichier MIDI a été enregistré avec succès
    if os.path.exists(output_midi_path):
        print(f'Le fichier MIDI a été enregistré avec succès sous : {output_midi_path}')
    else:
        print('Erreur lors de l\'enregistrement du fichier MIDI.')
        
    # Afficher le#s couleurs quantifiées
    plt.figure(figsize=(8, 6))
    plt.imshow([colors])
    plt.axis('off')
    plt.show()
    
# Importation de la bibliothèque Streamlit
import streamlit as st
from music21 import stream, note, converter, metadata
import subprocess
import os
import shlex

# ---------------------------------- Consignes pour run le code :
# Dans terminal windows (ou invité de commande), tapez:

# powershell
# cd D:\-PFE\ECE-2024-PFE      (nom du dossier où est enregistré le fichier)
# streamlit run application.py


#ls : pour voir le contenu du dossier
# cd : pour accéder à un chemin
# powershell : version + puissante du terminal

# -----------------------------------

st.image("./automne.jpg" )

# Ajout d'un logo
logo_path = "./logo2.png"
st.image(logo_path, width=200)

# Titre de la page
st.title("Bienvenue sur Picto Music")

# Sous-titre
st.header("Découvrez une nouvelle manière de composer de la musique")

# Paragraphe de texte
st.text("")

# Ajout des champ de texte :
titre_partition = st.text_input("Saisissez le titre de votre partition : ")
nom_compositeur = st.text_input("Saisissez votre nom de compositeur : ")
nom_audio = st.text_input("Saisissez le nom exact de votre fichier audio : ")
chemin_audio = f"./{nom_audio}.mid"

if nom_audio and not os.path.exists(chemin_audio):
    st.warning(f"Le fichier audio '{nom_audio}' n'existe pas. Veuillez vérifier le nom du fichier.")
    st.stop()


def generate_musescore_from_midi(midi_file_path, output_musicxml_path):
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
def main():

    # Ajout d'un bouton
    if st.button("Générer la partition"):
        # génération de la partition
        midi_file_path = f"./{nom_audio}.mid"
        output_musicxml_path = f"./partitions/{titre_partition}.musicxml"

        if not os.path.exists(output_musicxml_path):
            generate_musescore_from_midi(midi_file_path, output_musicxml_path)
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


# Appel de la fonction principale pour l'exécution de l'application
main()


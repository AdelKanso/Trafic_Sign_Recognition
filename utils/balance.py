import os
import random
from PIL import Image
import torchvision.transforms as T

# ======= CONFIGURATION GLOBALE =======
base_folder = r"C:\Users\samer\Desktop\Project_M2\ML_DL\DATA_N2\BelgiumTSC_Training\Training"  # <-- Chemin RACINE contenant les dossiers 00000 à 00061
target_size = 100                                                  # Nombre cible d'images PAR DOSSIER

# Rotation-only augmentation (entre -30° et +30°)
rotate_transform = T.RandomRotation(degrees=30)

# Définir les dossiers à traiter (de 00000 à 00061)
# Nous générons une liste de noms de dossiers formatés sur 5 chiffres.
folder_names = [f"{i:05d}" for i in range(62)] 

print(f"Dossiers à traiter : {len(folder_names)} dossiers.")

# ======= BOUCLE PRINCIPALE SUR TOUS LES DOSSIERS =======
total_augmented = 0

for folder_name in folder_names:
    folder_path = os.path.join(base_folder, folder_name)

    # 1. Vérification de l'existence du dossier
    if not os.path.exists(folder_path):
        print(f"⚠️ Dossier introuvable : {folder_path}. Ignoré.")
        continue

    # 2. Charger les images PPM existantes dans ce dossier
    # Le format PPM est souvent en minuscule ('.ppm')
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".ppm"))]

    current_count = len(images)
    
    # Si le dossier a déjà le nombre cible ou plus, on passe au suivant
    if current_count >= target_size:
        print(f"➡️ Dossier {folder_name}: Déjà {current_count} images. Poursuite.")
        continue

    print(f"\n🚀 Démarrage du dossier {folder_name} (Actuel: {current_count} images)")
    
    # 3. Boucle d'Augmentation
    images_to_add = target_size - current_count
    augmented_count_in_folder = 0

    while len(images) < target_size:
        # Choisir une image existante aléatoirement
        img_name = random.choice(images)
        img_path = os.path.join(folder_path, img_name)

        try:
            # Ouvrir l'image (PIL gère nativement le PPM)
            img = Image.open(img_path).convert("RGB")

            # Appliquer la rotation
            new_img = rotate_transform(img)

            # Sauvegarder la nouvelle image (nous utilisons .jpg pour éviter
            # que les fichiers augmentés soient aussi volumineux que les PPM originaux,
            # mais vous pouvez utiliser .ppm si vous y tenez vraiment)
            new_name = f"aug_{augmented_count_in_folder}_{folder_name}.jpg"
            new_img.save(os.path.join(folder_path, new_name))

            # Ajouter l'image nouvellement créée à la liste pour qu'elle puisse aussi
            # être choisie pour une augmentation future.
            images.append(new_name)
            augmented_count_in_folder += 1
            total_augmented += 1
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {img_name} dans {folder_name}: {e}")
            # Retirer l'image problématique pour éviter de boucler dessus
            images.remove(img_name)
    
    print(f"✅ Dossier {folder_name} terminé! Ajout de {augmented_count_in_folder} images. Total final: {len(images)}.")

print("\n---")
print(f"🎉 **PROCESSUS TERMINÉ**")
print(f"Nombre total d'images augmentées : {total_augmented}")
print("Chaque dossier devrait maintenant contenir au moins", target_size, "images.")
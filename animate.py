from PIL import Image
import glob
import os

# 1. Récupérer tous les fichiers du dossier et les trier par nom
image_folder = "images_vagues_structure"
search_path = os.path.join(image_folder, "*.png")
files = sorted(glob.glob(search_path))

# 2. Charger les images
frames = [Image.open(f) for f in files]

# 3. Sauvegarder en GIF
# duration=100 signifie 100ms entre chaque image (10 images/sec)
frames[0].save(
    "animation_vagues.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=200, 
    loop=0
)

print(f"Animation créée avec {len(frames)} images !")
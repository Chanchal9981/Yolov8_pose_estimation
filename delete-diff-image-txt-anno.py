from os import listdir,remove
from os.path import isfile, join

images_path = "full/path/to/folder_a"
annotations_path = "full/path/to/folder_b"


# this function will help to retrieve all files with provided extension in a given folder
def get_files_names_with_extension(full_path, ext):
    return [f for f in listdir(full_path) if isfile(join(full_path, f)) and f.lower().endswith(".{}".format(ext))]


images = get_files_names_with_extension(images_path, "jpg")
annotations = set([f.split(".")[0] for f in get_files_names_with_extension(annotations_path, "xml")])

for img in images:
    if img.split(".")[0] not in annotations:
        remove(join(images_path, img))

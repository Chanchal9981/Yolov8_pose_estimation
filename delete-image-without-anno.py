import os
import shutil
folderdir = "folder_path"
dirs = os.listdir(folderdir)
for i in dirs:
    if i.endswith(".txt"):
        txt = i
        jpg = str(i.split(".txt")[0]) + ".jpg"
        png = str(i.split(".txt")[0]) + ".png"
        jpeg = str(i.split(".txt")[0]) + ".jpeg"
        try:
            shutil.move(folderdir + "/" + jpg, 'dataset')
            shutil.move(folderdir + "/" + txt, 'dataset')
        
        except:
            try:
                shutil.move(folderdir + "/" + png, 'dataset')
                shutil.move(folderdir + "/" + txt, 'dataset')
            except:
                shutil.move(folderdir + "/" + jpeg, 'dataset')
                shutil.move(folderdir + "/" + txt, 'dataset')

print("DONE")

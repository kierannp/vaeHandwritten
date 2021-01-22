import glob
import os


classes = glob.glob('by_class/*')

for c in classes:
    for name in glob.glob(c+'/*.mit'):
        os.remove(name)
    for subdir in glob.glob(c+'/*'):
        for file in glob.glob(subdir+'/*'):
            split_name = file.split('/')
            del split_name[-2]
            new_name = '{}/{}/{}'.format(split_name[0],split_name[1],split_name[2])
            os.rename(file, new_name)
    for d in next(os.walk(c))[1]:
        os.rmdir('{}/{}'.format(c,d))
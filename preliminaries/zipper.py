import tarfile
import os

model_path = os.path.join('angel', 'models')
os.chdir(model_path)
print(os.getcwd())
print('Tarballing...')

with tarfile.open('models.tar.xz', 'w:xz') as tarball:
    for file in sorted(os.listdir(os.getcwd())):
        print(file)
        tarball.add(file)

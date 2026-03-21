import glob
import re

def remove_crops(p):
    with open(p, 'r') as f:
        c = f.read()
    c = re.sub(r' +crop_min:.*\n?', '', c)
    c = re.sub(r' +crop_max:.*\n?', '', c)
    with open(p, 'w') as f:
        f.write(c)

for p in glob.glob('experiments/*/configs/*.yaml'):
    remove_crops(p)
    

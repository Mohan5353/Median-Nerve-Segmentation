import glob
import os
import re

pat = re.compile(r"(\d+)\D*$")

def key_func(x):
    mat = pat.search(os.path.split(x)[-1])
    if mat is None:
        return x
    return "{:>10}".format(mat.group(1))

def check_dataset(data_path, min_frames=36):
    data_path = os.path.expanduser(data_path)
    all_patients = sorted([d for d in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(d)])
    
    short_videos = []
    
    for p in all_patients:
        found_dirs = glob.glob(os.path.join(p, '*_IMAGES'))
        if not found_dirs:
            if os.path.exists(os.path.join(p, 'images')):
                found_dirs = [p]
        
        for d in found_dirs:
            frames = sorted(glob.glob(os.path.join(d, 'images/*.jpg')), key=key_func)
            masks = sorted(glob.glob(os.path.join(d, 'masks/*.tif')), key=key_func)
            if not masks:
                masks = sorted(glob.glob(os.path.join(d, 'masks/*.png')), key=key_func)
            
            mask_indices = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}
            paired = [f for f in frames if os.path.splitext(os.path.basename(f))[0] in mask_indices]
            
            if len(paired) < min_frames:
                short_videos.append((d, len(paired)))
                
    return short_videos

if __name__ == "__main__":
    shorts = check_dataset("~/DATA-VisTr/")
    if shorts:
        print("Videos with less than 36 frames:")
        for path, count in shorts:
            print(f"{path}: {count} frames")
    else:
        print("No videos found with less than 36 frames.")

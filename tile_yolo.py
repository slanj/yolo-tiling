import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import argparse
import os
import random
from shutil import copyfile
 

def tiler(imnames, newpath, falsepath, slice_size, ext):
    for imname in imnames:
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = imname.replace(ext, '.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        
        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height
        
        boxes = []
        
        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w']/2
            y1 = (height - row[1]['y1']) - row[1]['h']/2
            x2 = row[1]['x1'] + row[1]['w']/2
            y2 = (height - row[1]['y1']) + row[1]['h']/2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
        
        counter = 0
        print('Image:', imname)
        # create tiles and find intersection with bounding boxes for each tile
        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                x1 = j*slice_size
                y1 = height - (i*slice_size)
                x2 = ((j+1)*slice_size) - 1
                y2 = (height - (i+1)*slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])        
                        
                        if not imsaved:
                            sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                            sliced_im = Image.fromarray(sliced)
                            filename = imname.split('/')[-1]
                            slice_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}{ext}')                            
                            slice_labels_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}.txt')                            
                            print(slice_path)
                            sliced_im.save(slice_path)
                            imsaved = True                    
                        
                        # get smallest rectangular polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope 
                        
                        # get central point for the new bounding box 
                        centre = new_box.centroid
                        
                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy
                        
                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size
                        
                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / slice_size
                        new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                        
                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                
                if len(slice_labels) > 0:
                    slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                    print(slice_df)
                    slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
                
                if not imsaved and falsepath:
                    sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                    sliced_im = Image.fromarray(sliced)
                    filename = imname.split('/')[-1]
                    slice_path = falsepath + "/" + filename.replace(ext, f'_{i}_{j}{ext}')                

                    sliced_im.save(slice_path)
                    print('Slice without boxes saved')
                    imsaved = True

def splitter(target, target_upfolder, ext, ratio):
    imnames = glob.glob(f'{target}/*{ext}')
    names = [name.split('/')[-1] for name in imnames]

    # split dataset for train and test

    train = []
    test = []
    for name in names:
        if random.random() > ratio:
            test.append(os.path.join(target, name))
        else:
            train.append(os.path.join(target, name))
    print('train:', len(train))
    print('test:', len(test))

    # we will put test.txt, train.txt in a folder one level higher than images

    # save train part
    with open(f'{target_upfolder}/train.txt', 'w') as f:
        for item in train:
            f.write("%s\n" % item)

    # save test part
    with open(f'{target_upfolder}/test.txt', 'w') as f:
        for item in test:
            f.write("%s\n" % item)

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-source", default="./yolosample/ts/", help = "Source folder with images and labels needed to be tiled")
    parser.add_argument("-target", default="./yolosliced/ts/", help = "Target folder for a new sliced dataset")
    parser.add_argument("-ext", default=".JPG", help = "Image extension in a dataset. Default: .JPG")
    parser.add_argument("-falsefolder", default=None, help = "Folder for tiles without bounding boxes")
    parser.add_argument("-size", type=int, default=416, help = "Size of a tile. Dafault: 416")
    parser.add_argument("-ratio", type=float, default=0.8, help = "Train/test split ratio. Dafault: 0.8")

    args = parser.parse_args()

    imnames = glob.glob(f'{args.source}/*{args.ext}')
    labnames = glob.glob(f'{args.source}/*.txt')
    
    if len(imnames) == 0:
        raise Exception("Source folder should contain some images")
    elif len(imnames) != len(labnames):
        raise Exception("Dataset should contain equal number of images and txt files with labels")

    if not os.path.exists(args.target):
        os.makedirs(args.target)
    elif len(os.listdir(args.target)) > 0:
        raise Exception("Target folder should be empty")

    # classes.names should be located one level higher than images   
    # this file is not changing, so we will just copy it to a target folder 
    upfolder = os.path.join(args.source, '..' )
    target_upfolder = os.path.join(args.target, '..' )
    if not os.path.exists(os.path.join(upfolder, 'classes.names')):
        print('classes.names not found. It should be located one level higher than images')
    else:
        copyfile(os.path.join(upfolder, 'classes.names'), os.path.join(target_upfolder, 'classes.names'))
    
    if args.falsefolder:
        if not os.path.exists(args.falsefolder):
            os.makedirs(args.falsefolder)
        elif len(os.listdir(args.falsefolder)) > 0:
            raise Exception("Folder for tiles without boxes should be empty")

    tiler(imnames, args.target, args.falsefolder, args.size, args.ext)
    splitter(args.target, target_upfolder, args.ext, args.ratio)
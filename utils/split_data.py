import os
import random

def split_data(dataset, test_ratio):
    # read all data
    img_dir_path = os.path.join('datasets', dataset, 'images')
    img_names = os.listdir(img_dir_path)

    # split data
    img_pathes = [os.path.join('./images', img_name) for img_name in img_names]
    test_images = random.sample(img_pathes, int(len(img_pathes)*test_ratio))
    train_images = [img_path for img_path in img_pathes if img_path not in test_images]

    # save datalist with txt file
    train_images = [img+'\n' for img in train_images]
    test_images = [img+'\n' for img in test_images]
    train_images[-1] = train_images[-1].strip('\n')
    test_images[-1] = test_images[-1].strip('\n')

    train_save_path = os.path.join('datasets', dataset, 'train.txt')
    test_save_path = os.path.join('datasets', dataset, 'val.txt')
    with open(train_save_path, "w") as file:
        file.writelines(train_images)
    with open(test_save_path, "w") as file:
        file.writelines(test_images)
    
    return train_images, test_images




if __name__=='__main__':
    split_data('seaships', 0.1)
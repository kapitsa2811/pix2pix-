from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import asarray
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from os import listdir
from PIL import Image
import numpy as np

# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def load_images(path,path1, size=(256, 256)):
    src_list, tar_list = list(), list()
    # enumerate filenames in directory, assume all are images
    for indx,filename in enumerate(listdir(path)):

        if indx<500:
            continue

        if indx%100==0:
            print("\n\t indx:::",indx)

        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)# load and resize the image
        pixels1 = load_img(path1 + filename, target_size=size)
        # convert to numpy array
        pixels1 = img_to_array(pixels1)

        # split into satellite and map

        sat_img, map_img = pixels, pixels1


        #print("\n\t sat_img=",sat_img.shape)
        src_list.append(sat_img)
        tar_list.append(map_img)

        if indx>700:
            break


    return [asarray(src_list), asarray(tar_list)]



# plot source, generated and target images
def plot_images(ix,src_img, gen_img, tar_img):


    import scipy.misc
    import cv2
    from skimage.transform import resize

    temp_src=src_img[0]
    #temp_src.resize((2592,3300))
    #print("\n\t src_img=",temp_src.shape)

    temp_src=(temp_src + 1) / 2.0

    temp_src=cv2.resize(temp_src,(2592,3300))
    #scipy.misc.imsave('.//prediction2//your_file.jpeg', temp_src)
    cv2.imwrite('.//prediction2//'+str(ix)+"_ori.jpg",255* temp_src)


    temp_gen_img=gen_img[0]

    temp_gen_img=(temp_gen_img + 1) / 2.0

    indThre=np.where(temp_gen_img>0.5)
    temp_gen_img[indThre]=1

    temp_gen_img=cv2.resize(temp_gen_img,(2592,3300))
    #scipy.misc.imsave('.//prediction2//your_file.jpeg', temp_src)

    cv2.imwrite('.//prediction2//'+str(ix)+"_GT2.jpg",255* temp_gen_img)


    temp_tar_img=tar_img[0]
    #temp_src.resize((2592,3300))
    #print("\n\t src_img=",temp_src.shape)

    temp_tar_img=(temp_tar_img + 1) / 2.0

    temp_tar_img=cv2.resize(temp_tar_img,(2592,3300))
    #scipy.misc.imsave('.//prediction2//your_file.jpeg', temp_src)
    cv2.imwrite('.//prediction2//'+str(ix)+"_GT.jpg",255* temp_tar_img)

    # im = Image.fromarray(src_img)
    #im.save("//prediction2//your_file.jpeg")

    #pyplot.savefig(".//prediction2//your_file.jpeg",src_img)
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Source', 'Generated', 'Expected']
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, 3, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show title
        pyplot.title(titles[i])
    #pyplot.show()


path = '/home/kapitsa/Documents/dataSets/facade2/train_label/'
path1="/home/kapitsa/Documents/dataSets/facade2/train_picture//"
# load dataset
[src_images, tar_images] = load_images(path,path1)


# load dataset
[X1, X2] = load_real_samples('lines_257.npz')
print('Loaded', X1.shape, X2.shape)
# load model
model = load_model('model_050200.h5')
# select random example
ix = randint(0, len(X1), 1)

print("\n\t ix=>>",ix)

import cv2

for ix in range(100):

    #src_image, tar_image = src_images[[ix]], tar_images[[ix]]

    src_image, tar_image = X1[[ix]], X2[[ix]]
    # generate image from source
    gen_image = model.predict(src_image)
    # plot all three images

    #print("\n\t type=",type(src_image))
    plot_images(ix,src_image, gen_image, tar_image)



    #src_image=cv2.resize(src_image,(2592,3300))

    # pyplot.savefig(".//prediction2//"+str(ix)+"_Orijpg",src_image)
    # pyplot.savefig(".//prediction2//"+str(ix)+"_Pred.jpg",gen_image)
    # pyplot.savefig(".//prediction2//"+str(ix)+"_GT.jpg",src_image)

    #pyplot.imwrite()
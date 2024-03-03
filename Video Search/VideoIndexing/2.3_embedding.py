import numpy as np
import keras
from keras import layers
from PIL import Image
from matplotlib import pyplot as plt

#size of encoded representation
#(52,52,1)
input_img = keras.Input(shape=(52,52,1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

#Model to map an input to its reconstruction
autoencoder= keras.Model(input_img, decoded)

#per-pixel binary crossentropy loss, and the Adam optimizer
optimizer= keras.optimizers.Adam(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

#Open file detection text
file=open("2.2_obj.txt", "r")
#extract data
infos=[]
for line in file:
    info=[]
    line= line.split("\t")
    for i in line:
        if "frame" in i:
            info.append(int(i.split(": ")[-1]))
        if "bboxinfo" in i:
            a=i.split(": ")[-1]
            b=a.split(", ")
            for dim in b:
                dim= dim.replace("[","").replace("]","")
                info.append(int(dim))        
            
    infos.append(info)

target_size= (52,52)

#Extract part of the frames where the objects are found
images=[]
for i in infos:
    image_path=f'Frames/{i[0]}.jpg'

    img=Image.open(image_path)
    img= img.convert('L')
    #Crop the frame
    xmin, xmax, ymin, ymax= i[1], i[2], i[3], i[4]

    crop= img.crop((xmin, ymin, xmax, ymax))
    reshape= crop.resize(target_size)

    image= np.array(reshape)/float(255)
    #image = cropped_image.flatten()
    #print(image.size)

    images.append(image)
#Final object images for training
final= np.array(images)
np.random.shuffle(final)
#print(final)
#Split the data
split= int(np.floor(len(final)*0.9))
x_train, x_test= np.split(final, [split], axis=0)
x_train= x_train[:,:,:,np.newaxis]
x_test= x_test[:,:,:,np.newaxis]

#print(x_train)
#print(x_test)

#Train autoencoder for 200 epochs
autoencoder.fit(x_train, x_train,
                epochs=200,
                batch_size=16,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

n = 20 # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(target_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(target_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

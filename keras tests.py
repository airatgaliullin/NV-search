from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rescale = 1./255,
vertical_flip=True,
zoom_range=0.2,
width_shift_range=0.2,


horizontal_flip = True)

img = load_img(r'D:\measuring\data\20180210\data\validate\NV\z=-3; x_c=0; y_c=-3; search_range=2um.png')

x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)


save_to_dir=r'D:\measuring\data\20180210\data\validate\preview'

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_to_dir, save_prefix='NV', save_format='png'):
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely
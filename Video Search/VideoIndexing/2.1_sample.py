from PIL import Image
import numpy as np

def sample_frame(start, end, step, target='L'):
    images=[]

    n= start
    while n<end:
        #get image
        file_path= f'./Frames/{n}.jpg'

        try:
            #open image from PIL
            with Image.open(file_path, 'r') as img:
                width, height= img.size
                #print(img.size)

                #resize the image by half
                resize= img.resize((int(width/2), int(height/2)))
                #print(f' {resize.size} \n')

                #GrayScale 
                if resize.mode != target:
                    final= resize.convert(target)

                #stored to show example of final image
                #final.save(f'./VideoOutPut/{n}.jpg')
                #Normalize
                img_array= np.array(final)/float(255)
                images.append(img_array)

        #exception error
        except FileNotFoundError:
            print(f'FileNotFound at {file_path}')
        except Exception as e:
            print(f'Error: {e}')

        n+=step

    #saves it to an npy file
    images_array= np.array(images)
    np.save('./2.1_frames.npy', images_array)

#sample frames from start to end with steps
fps=24
sample_frame(fps, fps*300, fps*1)


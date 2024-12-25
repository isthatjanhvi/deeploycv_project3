indices = [263,281]
layers_name = ['activation_6']
from IPython.display import image
for i in range(len(IMAGE_PATHS)):
    each_path = IMAGE_PATHS[i]
    index = indices[i]

    img = tf.keras.preprocessing.image.load_img(each_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    data = ([img],None)
    # define name with which to save result as
    name = each_path.split("/")[-1].split(".jpg")[0]
    #save the grad cam visullization
    explainer = GradCAM()
    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    grid = explainer.explain(data,model,'block5_conv3',index)
    explainer.save(grid,'.',name + 'gard_cam.png')
    display_images([each_path],name + 'grad_cam.png'])
cd /path/to/your/project

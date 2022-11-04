from flask import Flask, render_template, request


#Importing visualization functions
from visualization import *

#Importing the model
from model import *

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

model = generate_model()

#Generating directories
target_img = os.path.join(os.getcwd() , 'static/images')

img_dir = "photos_raw/"
mask_dir = "masks/"
test_mask_list = os.listdir(mask_dir)
test_image_list = os.listdir(img_dir)
test_image_list.sort()
test_mask_list.sort()


@app.route('/')
def index_view():
    return render_template('index.html', list = test_image_list)

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):

    img = load_img(filename, target_size=(256, 256))
    x = tf.keras.utils.img_to_array(img)
    y = np.expand_dims(x, axis=0)
    return y, x

@app.route("/predict" , methods=['GET', 'POST'])
def predict():
    
    image_name = request.form.get('file')
    file_path = img_dir + image_name
    mask_path = mask_dir + image_name

    img = resize(io.imread(file_path), (img_height, img_width))
    true_mask = convert_categories(io.imread(mask_path))
    rgb_mask, grayscale_mask = create_mask(model.predict(img[tf.newaxis,...]), img)

    image_path = os.path.join('static/images', 'photo.png')
    pred_mask_path = os.path.join('static/images', 'predicted_mask.png')
    true_mask_path = os.path.join('static/images', 'true_mask.png')

    plt.imsave(pred_mask_path, grayscale_mask.astype('uint8'))
    plt.imsave(image_path, img)
    plt.imsave(true_mask_path, true_mask)

    return render_template('predict.html', user_image = image_path, true_mask=true_mask_path,
        pred_mask=pred_mask_path, list = test_image_list)

@app.route('/custom',methods=['GET','POST'])
def custom():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', 'photo.png')
            file.save(file_path)
            img = resize(io.imread(file_path), (img_height, img_width))

            rgb_mask, grayscale_mask = create_mask(model.predict(img[tf.newaxis,...]), img)

            pred_mask_path = os.path.join('static/images', 'predicted_mask.png')
            plt.imsave(pred_mask_path, grayscale_mask.astype('uint8'))
            plt.imsave(file_path, img)
            

            return render_template('predict.html', user_image = file_path, 
                pred_mask=pred_mask_path, true_mask="none", list = test_image_list)
                
        else:
            return "Unable to read the file. Please check file extension"






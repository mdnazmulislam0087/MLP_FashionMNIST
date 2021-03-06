import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os



# create model 

def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS):
        
    LAYERS= [tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
            tf.keras.layers.Dense(units=300, activation="relu", name="hiddenLayer1"),
            tf.keras.layers.Dense(units=300, activation='relu', name='hiddenLayer2'),
            tf.keras.layers.Dense(units=10, activation='softmax', name='outputLayer')
             ]
    model_clf=tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)
    return model_clf


# To save model

def get_unique_filename(name):  #<< best it is
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{timestamp}_{name}"
    return unique_name

#save model
def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

## Save PLOTS
# save plot
# create unique path
def get_unique_filePlotname(name):  #<< best it is
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{timestamp}_{name}"
    return unique_name

def save_plot(df, plot_name, plots_dir ):
    unique_filename = get_unique_filePlotname(plot_name)
    path_to_plot = os.path.join(plots_dir, unique_filename)
    df.plot(figsize=(10, 7))
    plt.grid(True)
    #plt.show()
    plt.savefig(path_to_plot)

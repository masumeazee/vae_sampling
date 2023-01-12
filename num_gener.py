# -*- coding: utf-8 -*-
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

# Loading the iris dataset ...
iris = datasets.load_iris()
data = iris.data
target = iris.target

## Start the Class of Variational autoencoder
# Including encoder(intermediat layers, latent space) and decoder for sampling 
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            ## Input layer would be same the num of feature (in iris dataset --> 4)
            tf.keras.layers.Input(shape=(len(data[0]),)),
            ## Intermediate layer
            tf.keras.layers.Dense(64, activation='relu'),
            ## Latent layer like mu and log_var
            tf.keras.layers.Dense(2 * latent_dim)
        ])
        
        # same like the encoder layers here but in inverse laten --> intermediat --> original dim 
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(data[0]), activation='linear')
        ])
        
      #method of encoder output as mean(mu) and log_var in z latent space 
    def encode(self, x):
        z_mean, z_log_var = tf.split(self.encoder(x), 2, axis=-1)
        return z_mean, z_log_var
      
     # using reparameterising trick with eps vlaue
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
      
      # this phase of decoding from sampling to reconstructed data  
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits



# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# Create the model with laten dimension in VAE
latent_dim = 2 ## original dim
vae = VAE(latent_dim)

# Preprocess the data
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

# sampling phase with 100 samples in two step of reparameterising
# and generating samples with 2 laten dim

num_samples = 100
z = vae.reparameterize(tf.zeros((num_samples, latent_dim)),
                                      tf.zeros((num_samples, latent_dim)))

generated_samples = vae.decode(z, apply_sigmoid=False)

print(generated_samples.shape)
# Group similar samples together and
# Use any distance metric you prefer, such as L2 distance --> below
samples_distance = tf.norm(generated_samples - generated_samples[:, None], axis=-1)

#Define and Fit the Agglomerative model on the generated samples
agg_clustering = AgglomerativeClustering(n_clusters=3)
cluster_assignments = agg_clustering.fit_predict(generated_samples)

print(cluster_assignments)

# Create a scatter plot of the generated samples
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], c=cluster_assignments, cmap='rainbow', s=30)
plt.show

## __BARBA__ ##

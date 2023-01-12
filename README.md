# vae_sampling

Use #VAE for #sampling #numerical data based on entry dataset

#Variational #Autoencoders extend the core concept of #Autoencoder by placing 
constraints on how the identity map is learned. These constraints result in 
VAEs characterizing the lower-dimensional space, called the #latent_space, 
well enough that they are useful for data generation. VAEs characterize 
the latent space as a landscape of salient features seen in the training data,
rather than as a simple embedding space for data as AEs do.
They allow us to approximate high-dimensional latent spaces that can be sampled 
to generate new data.

In this project we use sampling of model for instance, to make new data from 
standard data of iris dataset then by utilizing any distance metric we measure
the distance between original data  and generated data.In next step there is some
different way to cluster the final matrix as result.   



More reviews could be found in the link below :

https://arxiv.org/abs/1312.6114

https://keras.io/examples/generative/vae/

https://openreview.net/forum?id=33X9fd2-9FyZd


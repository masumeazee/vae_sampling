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

 One way to generate new samples from a VAE is to use sampling methods, such as the
 reparameterization trick. This involves sampling from a random noise distribution 
(such as a standard normal distribution), and passing this noise through the VAE's 
decoder network to generate new samples. This can be useful for generating numeric datasets,
such as time series data or financial data.

In this project we use sampling model as a simple example, to make new data from 
standard data of #iris dataset then by utilizing any distance metric we measure
the distance between original data  and generated data.In next step there is some
different way to cluster the final matrix as result.   



More reviews can be found here, links below :

https://arxiv.org/abs/1312.6114    main paper

https://keras.io/examples/generative/vae/  standard code from keras blog

https://openreview.net/forum?id=33X9fd2-9FyZd  more reviews about vae model


from CLSWGAN import CLSWGANGP
wgan = CLSWGANGP()
wgan.train(epochs=30000, batch_size=1024, sample_interval=10)

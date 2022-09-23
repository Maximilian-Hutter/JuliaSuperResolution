function load_data(hparams)
    # #MLDatasets.MNIST.download(i_accept_the_terms_of_use=true)

    # # Load MNIST dataset
    # images, labels = MNIST(:train)[:]
    # # Normalize to [-1, 1]
    # image_tensor = reshape(@.(2f0 * images - 1f0), 28, 28, 1, :)
    # y = float.(Flux.onehotbatch(labels, 0:hparams.nclasses-1))
    # # Partition into batches
    # data = [(image_tensor[:, :, :, r], y[:, r]) |> gpu for r in partition(1:60000, hparams.batch_size)]
    return data
end
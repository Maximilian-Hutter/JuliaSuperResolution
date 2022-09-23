# function load_data(hparams)
#     # #MLDatasets.MNIST.download(i_accept_the_terms_of_use=true)
#     # # Load MNIST dataset
#     # images, labels = MNIST(:train)[:]
#     # # Normalize to [-1, 1]
#      image_tensor = reshape(@.(2f0 * images - 1f0), 640, 360, 3, :) # reshape to 640 360 and 3 channels
#     # y = float.(Flux.onehotbatch(labels, 0:hparams.nclasses-1))
#     # # Partition into batches
#      data = [(image_tensor[:, :, :, r], [:,:,:, r]) |> gpu for r in partition(1:60000, hparams.batch_size)]
#      #[:, :, :, r] shape of image
#     return data
# end

function get_data(args)
    ytrain = MLDatasets.MNIST(:train)[:]

    ytrain = reshape(ytrain, 2560, 1440, 3, :)
    ytest = reshape(ytest, 2560, 1440, 3, :)

    xtrain = reshape(xtest, 640, 360, 3, :)
    xtest = reshape(xtest, 640, 360, 3, :)

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=args.batchsize)
    
    return train_loader, test_loader
end
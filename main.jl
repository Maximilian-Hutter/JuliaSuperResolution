using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Images
using ImageMagick
using MLDatasets
using Statistics
using Parameters: @with_kw
using Random
using Printf
using CUDA
using Zygote

include("dataloader.jl")
include("utility.jl")
include("models.jl")  # include models in this line could also be placed in the middle of the file

@with_kw struct HyperParams
  batch_size::Int = 32
  img_size::Tuple{Int, Int, Int} = (640,360,3) # w, h, c # batch_size
  feats::Int = 64
  epochs::Int = 250
  resblocks::Int = 10
  kernel::Tuple{Int, Int} = (3,3)
  scale::Int = 2
  padding::Int = round((kernel[1] / 2))
  verbose_freq::Int = 1000  # steps until output of loss info and output img save
  lr::Float64 = 0.0002  # learning rate
end

function train_model(net, input, label, opt_net, hparams)
  # Random Gaussian Noise and Labels as input for the generator
  # noise = randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size)))
  # labels = rand(0:hparams.nclasses-1, hparams.batch_size)
  # y = Flux.onehotbatch(labels, 0:hparams.nclasses-1)
  # noise , y  = noise, float.(y) |> gpu

  # ps = Flux.params(gen.g_labels, gen.g_latent, gen.g_common)  # params important like that not whole model
  # loss = Dict()
  # loss["gen"], back = Zygote.pullback(ps) do
  #         fake = gen(y, noise)
  #         loss["discr"] = train_discr(discr, fake, y, original_data, label, opt_discr)
  #         generator_loss(discr(y, fake))
  # end
  # grads = back(1f0)
  # update!(opt_gen, ps, grads)
  # return loss

  y = float.(label) |> gpu  # check if changing label (mutation or side effects happening)

  ps = Flux.params(net)# full model params if it doesnt work use params(model.chain()) / params(model.nameOfParameters)
  loss = Dict()
  loss["net"], back = Zygote.pullback(ps) do
    generated = net(input)
    generator_loss(label, generated)
  end
  grads = back(1f0)
  update!(opt_net, ps, grads)
  return loss
end


function train(; kws...)
  hparams = HyperParams(kws...)

  # Load the data
  data = load_data(hparams)
  train_loader,test_loader = get_data(args)

  #fixed_labels = [float.(Flux.onehotbatch(rand(0:hparams.nclasses-1, 1), 0:hparams.nclasses-1)) |> gpu # onehot for categorical data
  
  # Generator
  net =  superResolution(hparams)

  # Optimizers
  opt_net = ADAM(hparams.lr, (0.5, 0.99))

  # Check if the `output` directory exists or needed to be created
  isdir("output")||mkdir("output")

  # Training
  train_steps = 0
  for ep in 1:hparams.epochs
      @info "Epoch $ep"
      for (x, y) in train_loader
          # Update discriminator and generator
          loss = train_net(net, x, y, opt_net, hparams)

          if train_steps % hparams.verbose_freq == 0
              @info("Train step $(train_steps), Model loss = $(loss["net"])")
              # Save generated fake image
              #output_image = create_output_image(net, fixed_noise, fixed_labels, hparams)
              #save(@sprintf("output/cgan_steps_%06d.png", train_steps), output_image)
          end

          train_steps += 1
      end
  end

  #output_image = create_output_image(gen, fixed_noise, fixed_labels, hparams)
  #save(@sprintf("output/cgan_steps_%06d.png", train_steps), output_image)
  #return Flux.onecold.(cpu(fixed_labels))
  return cpu(label)
end    

if abspath(PROGRAM_FILE) == @__FILE__
  train()
end
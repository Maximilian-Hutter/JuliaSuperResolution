using Flux
using CUDA
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
#using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
#using ProgressMeter: @showprogress

###################################################################
###################################################################
# Functions return the last value evaluated !!!!!!!!!!!!!
#   a = 10
#   b = 5
#   a + b
# end
# automatically returns 15 (last value)
###################################################################
###################################################################
###################################################################

# model
########################################################################
#TODO change filter sizes, change input, sizes, and change output sizes and correct all parameters

# structured like pytorch
# Res Skipped block
struct ResSkippedBlock  # like class in torch
  chain::Chain
end
function resSkippedBlock(args)  # buidling model
  return ResSkippedBlock(Chain(
    Conv(args.kernel, args.feats => args.feats, relu;pad = args.padding), # RELU

    Conv(args.kernel, args.feats => args.feats;pad = args.padding)
    ))
end
function (m::ResSkippedBlock)(x)    # forward path
    return (m.chain(x) * 0.1) # mul 0.1
end
Flux.@functor ResSkippedBlock
# Res Skipped Block end

# Upsample
struct UpsampleBlock
  chain::Chain 
end

function upsampleBlock(args)  # 4x upsample block
    return UpsampleBlock(Chain(
    if(args.scale & (args.scale - 1) == 0)    # if scale 2^n
        for _ in round(log(args.scale))   # repeat upscale for n
            Conv(args.kernel, args.feats => 4 * args.feats;pad = args.padding)
            PixelShuffle(2)
        end
    end

))
end

function (m::UpsampleBlock)(x)
    return m.chain(x)
end
Flux.@functor UpsampleBlock
#Upsample end

function repeatblocks(args)
    for _ in args.resblocks
        SkipConnection(resSkippedBlock(args),+)
    end
end

struct MainSkippedBlock
  chain::Chain
end
function mainSkippedBlock(args)  
  return MainSkippedBlock(Chain(# m not defined
    repeatblocks(args),  # for some reason cant put for loop in this line
    Conv(args.kernel, args.feats => args.feats;pad = args.padding)
    
    ))
#     for i in args.resblocks
#         SkipConnection(resSkippedBlock(args),+)
        
#     end

#     Conv(args.kernel, args.feats => args.feat;pad = args.padding)
#   ))
end

function (m::MainSkippedBlock)(x)
    return m.chain(x)
end
Flux.@functor MainSkippedBlock

struct SuperResolution
    chain::Chain
end

function superResolution(args)  #array start at 1

  return SuperResolution(Chain(
    # just for testing
    #conv
    Conv(args.kernel,3 => args.feats;pad = args.padding),
    SkipConnection(mainSkippedBlock(args), +),
         #res #conv in chain # input will be kept and added at end #skipped is the model & layers that are skipped     #add res value
    upsampleBlock(args),
    Conv(args.kernel, args.feats => args.feats; pad = args.padding)
  ))
end

function (m::SuperResolution)(x)
    return m.chain(x)
end
Flux.@functor SuperResolution
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
struct ResSkippedBlock  # like object in torch
  chain::Chain
end
function resSkippedBlock(args)  # buidling model
  return ResSkippedBlock(m.chain(
    Conv(filter, in => out, relu) # RELU
    Conv(filter, in => out)
    Flux.Scale()  # mul 0.1
  ))
end
function (m::ResSkippedBlock)(x)    # forward path
    return m.chain(x)
end
@functor ResSkippedBlock
# Res Skipped Block end

# Upsample
struct UpsampleBlock
  chain::Chain 
end

function upsampleBlock(args)  # 4x upsample block
  return UpsampleBlock(m.chain(
    Conv(filter, in => out)
    PixelShuffle(2)
    Conv(filter, in => out)
    PixelShuffle(2)
))
end

function (m::UpsampleBlock)(x)
    return m.chain(x)
end
@functor UpsampleBlock
#Upsample end

struct MainSkippedBlock
  chain::Chain
end
function mainSkippedBlock(args)
  return MainSkippedBlock(m.chain(
    ResSkippedBlock(x)  # TODO for loop for multiple blocks
    Conv(filter, in => out)
  ))
end

function (m::MainSkippedBlock)(x)
    return m.chain(x)
end
@functor MainSkippedBlock

struct SuperResolution
    chain::Chain
end

function superResolution(args)  #array start at 1

  return SuperResolution(Chain(
    # just for testing
    #conv
    filter = (5,5)
    firstPart = Chain(
    Conv(filter,3 => 7)
    SkipConnection(MainSkippedBlock(x), +))
         #res #conv in chain # input will be kept and added at end #skipped is the model & layers that are skipped     #add res value
    UpsampleBlock(firstPart)
    Conv(filter, in => out)
  ))
end

function (m::SuperResolution)(x)
    return m.chain(x)
end
@functor SuperResolution
#
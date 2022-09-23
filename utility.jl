function create_output_image(gen, fixed_noise, fixed_labels, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_labels, fixed_noise))
    @eval Flux.istraining() = true
    image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3, 4)), (2, 1))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end
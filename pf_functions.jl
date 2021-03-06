using Statistics
using Distributions
using StatsBase
using LinearAlgebra
using Plots
using Random

function create_uniform_particles(x_range, y_range, hdg_range, N)
	particles = zeros(N,3)
	particles[:, 1] = rand(Uniform(x_range[1], x_range[2]), N)
	particles[:, 2] = rand(Uniform(y_range[1], y_range[2]), N)
	particles[:, 3] = rand(Uniform(hdg_range[1], hdg_range[2]), N)
	particles[:, 3] .%= 2*pi
	
	return particles
end

function create_gaussian_particles(mean, std, N)
	particles = zeros(N,3)
	particles[:, 1] = rand(Normal(mean[1], std[1]), N)
	particles[:, 2] = rand(Normal(mean[2], std[2]), N)
	particles[:, 3] = rand(Normal(mean[3], std[3]), N)
	particles[:, 3] .%= 2*pi
	
	return particles
end

function predict(particles, u, std, seed=0, dt=1.)
	#move according to control input u tuple(heading change, velocity)
	#with noise Q (std heading change, std velocity)
	
	# number of rows for particles array
	N = size(particles, 1)
	
	# update heading 
	if seed != 0
		Random.seed!(seed)
	end
	particles[:, 3] .+= rand(Normal(u[1], std[1]), N)
	particles[:, 3] .%= 2 * pi
	
	# move in the (noisy) commanded direction
	if seed != 0
		Random.seed!(seed)
	end
	dist = (u[2] * dt) .+ (randn(N) * std[2])
	particles[:, 1] .+= cos.(particles[:, 3]) .* dist
	particles[:, 2] .+= sin.(particles[:, 3]) .* dist

	return particles
end

function update(particles, particle_weights, z, R, landmarks)
	updated_weights = particle_weights

	for (i, landmark) in enumerate(eachrow(landmarks))
		# for each landmark calculate the distance/norm for each particle to landmark
		distance = map(norm, eachslice(particles[:, 1:2] .- landmark' ,dims=1))
		# update particle weight by the probability of the observation to distance.
		updated_weights .*= [pdf(Normal(m,R), z[i]) for m in distance]
	end

	updated_weights .+= 1.e-300
	updated_weights ./= sum(updated_weights)

	return updated_weights
end

function estimate(particles, particle_weights_input)
	# return mean and variance of the weighted particles
	pos = particles[:, 1:2]
	mean_particles = mean(pos, weights(particle_weights_input), dims=1)
	var_particles = mean((pos .- mean_particles).^2, weights(particle_weights_input), dims=1)
	
	return mean_particles, var_particles
end

function simple_resample(particles, particle_weights)
	N = size(particles, 1)
	cumulative_sum = cumsum(particle_weights)
	# avoid round-off error
	cumulative_sum[lastindex(cumulative_sum)] = 1.
	# Using Ref, we keep cumulative_sum as an array for each broadcast step.
	indexes = searchsortedfirst.(Ref(cumulative_sum), rand(N))
	
	# resample according to indexes 
	particles = particles[indexes]
	fill!(particle_weights, 1. / N)
end

function neff(particle_weights)
	return 1. / sum(particle_weights .^2)
end

function systematic_resample(weights)
	N = size(weights)[1]

	# make N subdivisions, and choose positions with a consistent random offset
	positions = collect(range(0, step=1, length=N) .+ rand()) / N
	# zero array of type Int16
	indexes = zeros(Int16, N)
	
	# cumulative sum along the rows
	cumulative_sum = cumsum(weights, dims=1)
	i = 1
	j = 1
	while i < N + 1 
		if positions[i] < cumulative_sum[j]
			indexes[i] = j
			i += 1
		else  
			j += 1
		end
	end
	return indexes
end

function resample_from_index(particles, particle_weights, indexes)
	particles[:] = particles[indexes,:]
	weights_length = size(particle_weights)[1]
	particle_weights = fill(1/weights_length, (weights_length,))

	return particles, particle_weights
end
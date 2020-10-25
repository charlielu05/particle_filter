include("pf_functions.jl")

function run_pf()
    landmarks = [[-1 2]; [5 10]; [12 14]; [18 21]]
    N = 5000
    iters = 18
    sensor_std_err = .1 
    NL = size(landmarks)[1]
    xs = []
    ys = []
    actual_pos = []
    # create particles and weight
    particles = create_uniform_particles((0,20), (0,20), (0,6.28), N)
    particle_weights = ones(N) / N
    robot_pos = [0. 0.]	

    for i in 1:iters
        robot_pos += [1. 1.]
        # distance from robot to each landmark
        # these are measurements
        zs = map(norm, eachslice(landmarks .- robot_pos, dims=1)) .+ (randn(NL) * sensor_std_err)

        # move diagonally forward to (x+1, x+1)
        particles = predict(particles, (0., 1.414), (.2, .05))

        # incorporate measurements
        particle_weights = update(particles, particle_weights, zs, sensor_std_err, landmarks) 

        # resample if too few effective particles
        if neff(particle_weights) < N / 2
            indexes = systematic_resample(particle_weights)
            particles, particle_weights = resample_from_index(particles, particle_weights, indexes)
        end
        mu, var = estimate(particles, particle_weights)
        append!(xs, mu[1])
        append!(ys, mu[2])
        append!(actual_pos, [robot_pos])
    end
    return particles, particle_weights, xs, ys, actual_pos
end

particles_updated, weights_updated, xs, ys, actual_pos = run_pf()
#scatter(particles_updated[:,1], particles_updated[:,2])
scatter(xs, ys, label = "predicted")
scatter!(getindex.(actual_pos, 1), getindex.(actual_pos, 2), label = "actual")

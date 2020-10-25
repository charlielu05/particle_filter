using Test
include("pf_functions.jl")

function foo(x)
    return length(x)^2
end

function check_foo()
    @test foo("bar") == 9
    println("Foo checked")
end

# check uniform_particles function
function check_uniform_particles()
    N = 10
    Random.seed!(1234)
    uniform_particles = create_uniform_particles((0,1), (0,1), (0, pi*2), N)
    @test size(uniform_particles)[1] == N
    @test abs(0.5 - mean(uniform_particles[:,1])) < 0.15
    @test abs(0.5 - mean(uniform_particles[:,2])) < 0.15
    println("uniform particles checked")
end

# check predict function
function check_predict_function()
    # set random seed to fixed 
    rand_seed = 1234
    Random.seed!(rand_seed)
    N = 2
    test_predict_mu = (1., 1.5)
    test_predict_std = (.1, .05)
    # generate a fixed seed test uniform particles
    #uniform_particles = create_uniform_particles((0,1), (0,1), (0, pi*2), N)
    uniform_particles = [.5 .5 3; .1 .1 2]
    predict_particles = predict(uniform_particles, test_predict_mu, test_predict_std, rand_seed)
    @test predict_particles == [-0.4038387214269562 -0.7510229309903971 4.086734720195125; 
                                -1.3160113953515729 0.43419008158465533 2.9098256184143185]
end

# check update function
function check_update_function()
    test_pos = [-.5 -.5]
    test_array = [[-.5 -.5];[-.6 -.4];[10 10];[100 100]]
    test_landmarks = [[-.5 -.5];[-6 -.5];[10 10];[-10 -10]]
    test_weights = ones(size(test_array)[1]) / size(test_array)[1]
    zs = [.035 5.587 14.906 13.59]
    NL = size(test_landmarks)[1]
    updated_weights = update(test_array, test_weights, zs, .1, test_landmarks)
    @test updated_weights == [0.8632380238609324, 0.13676197613906765, 8.25969534731607e-302, 8.25969534731607e-302]
end

function check_estimate_function()
    Random.seed!(1234)
    test_particles = create_uniform_particles((0,1), (0,1), (0,5), 1000)
    test_weights = ones(1000) .* .25
    estimate_particle, _ = estimate(test_particles, test_weights)
    @test estimate_particle == [0.49684883432553845 0.5030967587104654]
end

function check_neff_function()
    @test round(neff([1 2 2]), digits=3) == 0.111
end 

function check_systematic_resample()
    test_weight = range(0, stop=1, step=.2)
    @test systematic_resample(test_weight) == [2,3, 3, 4, 4, 4]
end

function check_resample_from_index()
    particles = [[1 1]; [2 2]; [3 3]; [4 4]]
    particle_weights = [.9, .9, .9, .9]
    index = [3, 3, 3, 3]
    
    particles, particle_weights= resample_from_index(particles, particle_weights, index)
    @test particles == [[3 3]; [3 3]; [3 3]; [3 3]]
    @test particle_weights == [.25, .25, .25, .25]
end

check_foo()
check_uniform_particles()
check_predict_function()
check_update_function()
check_estimate_function()
check_neff_function()
check_systematic_resample()
check_resample_from_index()
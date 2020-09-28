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
    uniform_particles = create_uniform_particles((0,1), (0,1), (0, pi*2), N)
    @test size(uniform_particles)[1] == N
    @test abs(0.5 - mean(uniform_particles[:,1])) < 0.15
    @test abs(0.5 - mean(uniform_particles[:,2])) < 0.15
    println("uniform particles checked")
end

# check predict function
function check_predict_function()
    # set random seed to fixed 
    Random.seed!(1234)
    x1 = rand(2)
    Random.seed!(1234)
    x2 = rand(2)
    @test x1 == x2
    println("predict tested")
end


check_foo()
check_uniform_particles()
check_predict_function()
from pgpelib import PGPE
import numpy as np

pgpe = PGPE(
    solution_length=5,   # A solution vector has the length of 5
    popsize=20,          # Our population size is 20

    #optimizer='clipup',          # Uncomment these lines if you
    #optimizer_config = dict(     # would like to use the ClipUp
    #    max_speed=...,           # optimizer.
    #    momentum=0.9
    #),

    #optimizer='adam',            # Uncomment these lines if you
    #optimizer_config = dict(     # would like to use the Adam
    #    beta1=0.9,               # optimizer.
    #    beta2=0.999,
    #    epsilon=1e-8
    #),

)

# Let us run the evolutionary computation for 1000 generations
for generation in range(1000):

    # Ask for solutions, which are to be given as a list of numpy arrays.
    # In the case of this example, solutions is a list which contains
    # 20 numpy arrays, the length of each numpy array being 5.
    solutions = pgpe.ask()

    # This is the phase where we evaluate the solutions
    # and prepare a list of fitnesses.
    # Make sure that fitnesses[i] stores the fitness of solutions[i].
    fitnesses = [...]  # compute the fitnesses here

    # Now we tell the result of our evaluations, fitnesses,
    # to our solver, so that it updates the center solution
    # and the spread of the search distribution.
    pgpe.tell(fitnesses)

# After 1000 generations, we print the center solution.
print(pgpe.center)
import pickle
import pylab

with open('output.pickle', 'rb') as f:
    output = pickle.load(f)

print(output.keys())
for key, val in output.items():
    try:
        print(key, type(val), len(val), val.shape)
    except:
        print(key, type(val), len(val), val[0].shape)
print(output["population"][:10,:])

# pylab.scatter(output['population_energies'][:,0], output['population_energies'][:,1])
# pylab.show()


# def dominates(x, y):
#     return all(x_i <= y_i for x_i, y_i in zip(x, y)) and any(x_i < y_i for x_i, y_i in zip(x, y))

# with open("output.pickle", "rb") as f:
#     data = pickle.load(f)

# population_energies = data["population_energies"]

# # Find the non-dominated solutions
# pareto_front = []
# for i, solution in enumerate(population_energies):
#     is_dominated = False
#     for j, other_solution in enumerate(population_energies):
#         if i != j and dominates(other_solution, solution):
#             is_dominated = True
#             break
#     if not is_dominated:
#         pareto_front.append(solution)

# # Print or process the solutions in the Pareto front
# for solution in pareto_front:
#     print(solution)
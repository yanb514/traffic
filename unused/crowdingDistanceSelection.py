
import  numpy

class CrowdingDistanceSelection:
    def __init__(self, population):
        self.population = population


    def get_rankings(self):
        remaining = [i for i in range(self.population.shape[0])]

        current_rank = 0

        result = {}

        p = self.population.shape[1]

        while len(remaining) > 0:
            not_dominated = []
            for ind_a in remaining:

                dominated_by_one = False

                for ind_b in remaining:
                    dominated = True
                    if ind_b == ind_a:
                        continue

                    for i in range(p):
                        if self.population[ind_a, i] <= self.population[ind_b, i]:
                            dominated = False
                            break

                    if dominated:
                        dominated_by_one = True

                if not dominated_by_one:
                    not_dominated.append(ind_a)

            result[current_rank] = not_dominated

            for el in not_dominated:
                remaining.remove(el)

            current_rank = current_rank + 1

        return result

    def get_crowding_distance(self, indices):
        distances = {ind:0 for ind in indices}
        # p = self.population.shape[1]

        for p in range(self.population.shape[1]):
            order = sorted(indices, key=lambda ind:self.population[ind,p])

            f_m_min = min(self.population[:,p])
            f_m_max = max(self.population[:,p])

            distances[order[0]] = numpy.inf
            distances[order[-1]] = numpy.inf

            d = f_m_max-f_m_min

            for j in range(1,len(order)-1):
                distances[order[j]] = distances[order[j]] + (self.population[order[j+1],p]-self.population[order[j-1],p])/d

        result = []

        for el in indices:
            result.append(distances[el])

        indices_to_ret = sorted(indices, key=lambda ind:distances[ind], reverse=True)

        return result,indices_to_ret
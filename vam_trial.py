from collections import defaultdict
import numpy
import sys
import scipy
##########################

def find_cycle(m, n, basis):
    basis = basis.copy()
    rows, cols = [0]*m, [0]*n
    for i, j in basis:
        rows[i] += 1
        cols[j] += 1
    while True:
        is_ex = True
        for k in range(m):
            if rows[k] == 1:
                is_ex = False
                for i, j in basis:
                    if i == k:
                        cols[j] -= 1
                        rows[i] = 0
                        basis.remove((i, j))
                        break
        for k in range(n):
            if cols[k] == 1:
                is_ex = False
                for i, j in basis:
                    if j == k:
                        rows[i] -= 1
                        cols[j] = 0
                        basis.remove((i, j))
                        break
        if is_ex:
            return basis
        if len(basis) < 4:
            return None


def find_potentials(C, basis):
    inf = float('inf')
    u = [inf]*len(C)
    v = [inf]*len(C[0])
    u[0] = 0
    for _ in range(len(basis)):
        for i, j in basis:
            if v[j] == inf and u[i] != inf:
                v[j] = C[i][j] - u[i]
                break
            elif u[i] == inf and v[j] != inf:
                u[i] = C[i][j] - v[j]
                break
    return u, v


def split_cycle(i0, j0, cycle):
    neg, pos = set(), set()
    pos.add((i0, j0))
    for _ in range(len(cycle) >> 1):
        for i, j in cycle:
            if i == i0 and j != j0:
                neg_i, neg_j = i, j
                break
        neg.add((neg_i, neg_j))
        for i, j in cycle:
            if j == neg_j and i != neg_i:
                i0, j0 = i, j
                break
        pos.add((i0, j0))
    return neg, pos


def solve(a, b, C,X,basis,no_basis):
    m, n = len(a), len(b)
    diff = sum(a) - sum(b)
    if diff < 0:
        a.append(abs(diff))
        m += 1
        C.append([0 for _ in range(n)])
    elif diff > 0:
        b.append(diff)
        n += 1
        for row in C:
            row.append(0)
    
    if len(no_basis) == 0:
        return X

    while True:
        u, v = find_potentials(C, basis)
        i0, j0 = min(no_basis, key=lambda x: C[x[0]][x[1]]-u[x[0]]-v[x[1]])
        min_delta = C[i0][j0]-u[i0]-v[j0]
        if min_delta >= 0:
            return X

        basis.add((i0, j0))
        no_basis.remove((i0, j0))
        cycle = find_cycle(m, n, basis)
        neg, pos = split_cycle(i0, j0, cycle)
        i_star, j_star = min(neg, key=lambda el: X[el[0]][el[1]])
        theta = X[i_star][j_star]
        for el in pos:
            X[el[0]][el[1]] += theta
        for el in neg:
            X[el[0]][el[1]] -= theta
        basis.remove((i_star, j_star))
        no_basis.add((i_star, j_star))

#############################

r,c=input().split(' ')
col_ind=[]
row_ind=[]
for i in range(int(c)):
    #col_ind.append(str(chr(65+int(i))))
    col_ind.append('D'+str(int(i)+1))
for i in range(int(r)):
    #row_ind.append(str(chr(77+int(i))))
    row_ind.append('O'+str(int(i)+1))

trial_dict = [dict() for x in range(int(r))]
current_net_cost_matrix={}
temp_list=[]
trial_list=[]
another_costs=[]
for i in range(int(r)):
    temp_list=input().split(' ')
    k=0
    for j in range(int(c)):
        trial_dict[i][col_ind[j]]=int(temp_list[j])
        trial_list.append(int(temp_list[j]))
    current_net_cost_matrix[row_ind[i]]=trial_dict[i]
    another_costs.append(trial_list)
    trial_list=[]
    temp_list.clear()
#print(another_costs)   
#current_net_cost_matrix  = {'W': {'A': 16, 'B': 16, 'C': 13, 'D': 22, 'E': 17},
#          'X': {'A': 14, 'B': 14, 'C': 13, 'D': 19, 'E': 15},
#          'Y': {'A': 19, 'B': 19, 'C': 20, 'D': 23, 'E': 50},
#          'Z': {'A': 50, 'B': 12, 'C': 50, 'D': 15, 'E': 11}}

# current_net_cost_matrix  = {'W': {'A': 3, 'B': 1, 'C': 7, 'D': 4},
#           'X': {'A': 2, 'B': 6, 'C': 5, 'D': 9},
#           'Y': {'A': 8, 'B': 3, 'C': 3, 'D': 2},
#           }


#current_supply_tp = {'W': 50, 'X': 60, 'Y': 50, 'Z': 50}
#current_supply_tp = {'O1': 300, 'O2': 400, 'O3': 500}
current_supply_tp={}
another_supply=[]
temp_list=input().split(' ')
for i in range(int(r)):
    current_supply_tp[row_ind[i]]=int(temp_list[i])
    another_supply.append(int(temp_list[i]))
temp_list.clear()

#current_demand_tp = {'A': 30, 'B': 20, 'C': 70, 'D': 30, 'E': 60}
#current_demand_tp = {'A': 250, 'B': 350, 'C': 400, 'D': 200}

current_demand_tp={}
another_demand=[]
temp_list=input().split(' ')
i=0
for i in range(int(c)):
    current_demand_tp[col_ind[i]]=int(temp_list[i])
    another_demand.append(int(temp_list[i]))
temp_list.clear()
cols = sorted(current_demand_tp.keys())

res = dict((k, defaultdict(int)) for k in current_net_cost_matrix)
g = {}
for x in current_supply_tp:
    g[x] = sorted(current_net_cost_matrix[x].keys(), key=lambda g: current_net_cost_matrix[x][g])
for x in current_demand_tp:
    g[x] = sorted(current_net_cost_matrix.keys(), key=lambda g: current_net_cost_matrix[g][x])
 
while g:
    d = {}
    for x in current_demand_tp:
        d[x] = (current_net_cost_matrix[g[x][1]][x] - current_net_cost_matrix[g[x][0]][x]) if len(g[x]) > 1 else current_net_cost_matrix[g[x][0]][x]
    s = {}
    for x in current_supply_tp:
        s[x] = (current_net_cost_matrix[x][g[x][1]] - current_net_cost_matrix[x][g[x][0]]) if len(g[x]) > 1 else current_net_cost_matrix[x][g[x][0]]
    f = max(d, key=lambda n: d[n])
    t = max(s, key=lambda n: s[n])
    t, f = (f, g[f][0]) if d[f] > s[t] else (g[t][0], t)
    v = min(current_supply_tp[f], current_demand_tp[t])
    res[f][t] += v
    current_demand_tp[t] -= v
    if current_demand_tp[t] == 0:
        for k, n in current_supply_tp.items():
            if n != 0:
                g[k].remove(t)
        del g[t]
        del current_demand_tp[t]
    current_supply_tp[f] -= v
    if current_supply_tp[f] == 0:
        for k, n in current_demand_tp.items():
            if n != 0:
                g[k].remove(f)
        del g[f]
        del current_supply_tp[f]
basis_check=[]
cost_final_matrix=[]
basis = set()
no_basis=set()

for n in cols:
    print ("\t", n,end="")
print("\n")
cost = 0
for g in sorted(current_net_cost_matrix):
    trial=[]
    temp_matrix=[]
    print (g, "\t",end="")
    for n in cols:
        trial=[]
        y = res[g][n]
        temp_matrix.append(int(y))
        if y != 0:
            trial.append(int(g[-1])-1)
            trial.append(int(n[-1])-1)
            basis.add((int(g[-1])-1, int(n[-1])-1))
            basis_check.append(trial)
            print (y,end=" ")
        else:
            no_basis.add((int(g[-1])-1, int(n[-1])-1))
        cost += y * current_net_cost_matrix[g][n]
        print ("\t",end="")
    print("\n")
    cost_final_matrix.append(temp_matrix)

print ("\n\nTotal Cost = ", cost)
print(basis_check)
# print("Final Matrix is :")
# print(cost_final_matrix)
print(basis)
print(no_basis)
 
X = solve(another_supply, another_demand, another_costs,cost_final_matrix,basis,no_basis)
final_ans=0;
for i in range(len(X)):
    for j in range(len(X[i])):
        final_ans+=X[i][j]*another_costs[i][j]
# for row_index, row in enumerate(X):
#     for col_index, item in enumerate(X):
#         final_ans+=X[row_index][col_index]*another_costs[row_index][col_index]     
        
for row in X[:int(r)]:
    print(*row[:int(c)])
print(final_ans)




   
import networkx as nx
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import os
import pickle


def pathLength(sg,s,v):
    try:
        return nx.shortest_path_length(sg, s, v)
    except:
        #5M should be enough i can aford to return max int since this numer can be multiplied
        return -1


def generateSubraph(original_graph, seeds, nodes_per_seed):
    rg = nx.Graph()
    overlap_counter = 0
    for s in seeds:
        try:
            i = 0
            for e in nx.bfs_edges(original_graph, s):
                if not rg.has_node(e[0]):
                    i += 1
                if not rg.has_node(e[1]):
                    i += 1
                if i > nodes_per_seed:
                    print('seed done')
                    break
                rg.add_edge(e[0], e[1])
                rg.get_edge_data(e[0], e[1]).update(original_graph.get_edge_data(e[0], e[1]))

        except:
            e = sys.exc_info()[0]
            print(s)
    return rg


def find_two_distnat_clusters(g,min_distance,min_degree):
    while True:
        a = np.random.choice(g.nodes, 1)[0]
        while True:
            a = np.random.choice(g.nodes, 1)[0]
            try:
                if a and nx.degree(g, a) >= min_degree:
                    break
            except:
                continue

        for pb in g.nodes:
            if not isinstance(pb, str):
                continue
            degree = nx.degree(g, pb)
            if degree >= min_degree:
                dist = pathLength(g, a, pb)
                if dist >= min_distance:
                    print("d:" + str(dist) + " " + str(a) + ":" + str(pb) + " ad:" + str(nx.degree(g, a)) + " bd: " + str(degree))
                    return [a, pb]



def dist_matrix(g):
    nodes = {}
    dist = nx.shortest_path(g)
    for a in dist.keys():
       if a not in nodes:
           nodes[a] = {}
       for b in dist[a].keys():
            if a!=b:
                if b not in nodes:
                    nodes[b] = {}
                d = len(dist[a][b])
                nodes[a][b] =  d
                nodes[b][a] = d
            else:
                nodes[b][a] = 0
    return nodes

def get_average_per_node_tuples(g,matrix):
    ret = {}
    for v in g:
        sum = 0
        cnt = 0
        if v in matrix:
            for n in matrix[v].values():
                if n >0:
                    sum+=n
                    cnt+=1
            ret[v]=sum/cnt
    return ret


def get_average_bacon_number(matrix):
    sum = 0
    counter = 0
    for v in matrix.values():
        for w in v.values():
            if w > 0:
                sum+=w
                counter+=1
    return sum/counter

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def compute_and_store_histogram(graph,attribute,name=None,buckets=20,normed=False):
    plt.title(str(attribute) + ' histogram')
    plt.xlabel(attribute)

    data =[graph.node[n][attribute] for n in graph.nodes]
    compute_and_store_histogram_arr(data,attribute,name=name,buckets=buckets,normed=normed)
    return

def compute_and_store_histogram_arr(data,title,name=None,buckets=20,normed=False):
    plt.title(str(title) + ' histogram')
    binwidth = (max(data)-min(data))/buckets
    bins = [b for b in frange(min(data), max(data) + binwidth, binwidth)]
    plt.hist(data, bins=bins,normed=normed)
    if name is not None:
        plt.savefig(name)
    plt.clf()
    plt.cla()
    return

def get_top_n(dict,n,rev=True):
    return sorted(dict.items(), key=itemgetter(1), reverse=rev)[0:n]

def pritn_top_n(dict,n,rev=True):
    for v in get_top_n(dict,n,rev=rev):
        print(v)


##load graph filter out suspicious values


g = nx.Graph()
if os.path.exists('graph.pckl'):
    g = nx.read_gpickle('graph.pckl')
else:
    df = pd.read_csv('casts.csv', sep=';', names=["ID", "MOVIE", "ACTOR", "J", "ROLE"])
    df = df[(df.ACTOR != "s a") & (df.ACTOR != None) & (df.ACTOR != "")]
    for movie in df.iloc[:, 1].unique():
        cast = list(df[df.iloc[:, 1] == movie].iloc[:, 2])
        for i, item in enumerate(cast):
            for j in range(i + 1, len(cast)):
                edge = g.get_edge_data(cast[i], cast[j])
                if edge is not None:
                    edge['count'] += 1
                else:
                    g.add_edge(cast[i], cast[j], count=1)
    nx.write_gpickle(g, 'graph.pckl')

print("graph contains " + str(len(list(nx.bridges(g))))+ " bridges")
print('whole graph has:' + str(g.number_of_nodes()))
print('Highest degrees')
pritn_top_n(dict(g.degree()),5)

print("Most costarings")
edges = dict()
for a,b,c in  g.edges.data('count', default=0):
    edges[(a,b)]=c

pritn_top_n(edges,5)

print("Most costarings filtered")
edges = dict()
for a,b,c in  g.edges.data('count', default=0):
    if a != b:
        edges[(a,b)]=c
pritn_top_n(edges,5)



components = [c.number_of_nodes() for c in nx.connected_component_subgraphs(g)]
compute_and_store_histogram_arr(components,'componnes sizes','fig/component_size_dist',buckets=20)

s_compoennts = sorted(components,reverse=True)
s_compoennts = s_compoennts[1:]
compute_and_store_histogram_arr(s_compoennts,'componnes sizes','fig/component_size_dist_without_main',buckets=20)


print('graph contains:' + str(len(components)) + ' disconnected components')
nx.set_node_attributes(g, dict(g.degree()), name='degree')
compute_and_store_histogram(g, 'degree', buckets=600, name='fig/degree_histogram_whole')
compute_and_store_histogram(g, 'degree', buckets=20, name='fig/degree_histogram_whole_20b')

seed = ['Humphrey Bogart','Jack Nicholson','Donald meek','Carol Burnett']
if os.path.isfile('subgraph.gefx'):
    rg = nx.read_gexf('subgraph.gefx')
else:
    seed=seed + find_two_distnat_clusters(g,4,25)
    rg = generateSubraph(g,seed,500)

    print('graph contains:' + str(sum(1 for c in nx.connected_component_subgraphs(rg))) + ' disconnected components')
    print("Subgraph contains: " + str(rg.number_of_nodes()))


    lc = nx.load_centrality(rg)
    nx.set_node_attributes(rg,lc,name='load_centrality')
    compute_and_store_histogram(rg,'load_centrality','fig/subgrpah_load_cetrality_hist',buckets=10)
    for e in rg.edges:
        ed = rg.get_edge_data(e[0], e[1])
        ed['wl'] =1000*(lc[e[0]]+lc[e[1]])

    #compute degree of subgraph
    nx.set_node_attributes(rg,dict(rg.degree()),name='degree')
    compute_and_store_histogram(rg,'degree',name='subgraph_degree_hist')


    #compute bacon numbers get best and worst
    bacon_matrix = dist_matrix(rg)
    averagee_bacon = get_average_bacon_number(bacon_matrix)

    print(averagee_bacon)
    bacon_dict = get_average_per_node_tuples(rg,bacon_matrix)

    print('Best bacon numbers')
    pritn_top_n(bacon_dict,5,rev=False)
    print('Worst bacon numbers')
    pritn_top_n(bacon_dict,5,rev=True)

    nx.set_node_attributes(rg,bacon_dict,name='bacon')
    compute_and_store_histogram(rg,'bacon','fig/subgraph_bacon_hist',buckets=60)
    nx.write_gexf(rg,'subgraph.gefx')

#COMPUTE REUSABLE POSSITION
if os.path.isfile('pos.pickle'):
    with open('pos.pickle', 'rb') as f:
        pos_a = pickle.load(f)
else:
    pos_a=nx.spring_layout(rg,iterations=50, scale=0.05, weight='wl')
    with open('pos.pickle', 'wb') as f:
        pickle.dump(pos_a, f)

distances = [min([pathLength(rg, s, v) for s in seed]) for v in rg]
sizes = [max([600 -(500*(d)),5]) for d in distances]
labels = {}
for s in seed:
    labels[s] = s

plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
nx.draw_networkx(rg, pos=pos_a, arrows=False, with_labels=True, node_size=sizes, width=0.2, style='dashed', cmap = plt.get_cmap("viridis"), node_color=[d for d in distances],labels=labels)
plt.show()
plt.savefig('fig/initial_graph_with_seeds')
plt.clf()
plt.cla()

lc = [(n,rg.node[n]['load_centrality']) for n in rg.nodes]
ord_lc=sorted(lc,key=itemgetter(1),reverse=True)
labels = {}
for k,v in ord_lc[0:5]:
    labels[k]=k

sizes = [3000*rg.node[n]['load_centrality'] for n in rg.nodes]
colors = [3000*rg.node[n]['load_centrality'] for n in rg.nodes]
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
nx.draw_networkx(rg, pos=pos_a,arrows=False, with_labels=True,node_size=[3000*lc[n] for n in rg.nodes],width=0.1,style='dashed',cmap = plt.get_cmap("viridis_r"),node_color=[3000*lc[n] for n in rg.nodes],labels=labels )
fig.savefig('fig/eigenvalues')
fig.clear()



averagee_bacon = sum([rg.node[d]['bacon'] for d in rg.nodes])/rg.number_of_nodes()
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
nx.draw_networkx(rg, pos=pos_a, arrows=False, with_labels=True, node_size=[200 + ((averagee_bacon-rg.node[d]['bacon'])*50) for d in rg.nodes], width=0.1, style='dashed', cmap = plt.get_cmap("viridis_r"), node_color=[(averagee_bacon-rg.node[d]['bacon']) for d in rg.nodes],label=seed)
fig.savefig('fig/eigenvalues')
fig.clear()




eigenvector_centrality = nx.eigenvector_centrality(rg,max_iter=300)
ord_lc=sorted(eigenvector_centrality.items(),key=itemgetter(1),reverse=True)
labels = {}
for k,v in ord_lc[0:5]:
    labels[k]=k


fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
nx.draw_networkx(rg, pos=pos_a,arrows=False, with_labels=True,node_size=[(1000*eigenvector_centrality[v])+20 for v in rg.nodes],width=0.1,style='dashed',cmap = plt.get_cmap("viridis_r"),node_color=[eigenvector_centrality[v] for v in rg.nodes],labels=labels)
fig.savefig('fig/eigenvalues')
fig.clear()



communities = nx.clustering(rg)
ord_lc=sorted(communities.items(),key=itemgetter(1),reverse=True)
labels = {}
for k,v in ord_lc[0:15]:
    labels[k]=k
trg = rg.copy()
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
nx.draw_networkx(trg, pos=pos_a,arrows=True, with_labels=True,node_size=[200*communities[v]+20 for v in rg],width=0.1,style='dashed',cmap = plt.get_cmap("viridis"),node_color=[communities[v] for v in rg],labels=labels )
fig.savefig('fig/communities')
fig.clear()



communities = nx.square_clustering(rg)
ord_lc=sorted(communities.items(),key=itemgetter(1),reverse=True)
labels = {}
for k,v in ord_lc[0:5]:
    labels[k]=k
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
nx.draw_networkx(rg, pos=pos_a,arrows=False, with_labels=False,node_size=[200*communities[v]+20 for v in rg],width=0.1,style='dashed',cmap = plt.get_cmap("viridis"),node_color=[communities[v] for v in rg] )
fig.savefig('fig/communities_square')
fig.clear()


bridges = [b for b in nx.bridges(rg)]
def getBridgeSize(bridges,node):
    if node in bridges:
        return 250
    else:
        return 25

bridge_sizes = [getBridgeSize(bridges,n) for n in rg]
fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
nx.draw_networkx(rg, pos=pos_a,arrows=False, with_labels=False,node_size=bridge_sizes,width=0.1,style='dashed',cmap = plt.get_cmap("viridis"),node_color=bridge_sizes )
fig.savefig('fig/bridges')
fig.clear()



print()




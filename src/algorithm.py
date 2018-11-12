import networkx as nx
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def nodemapping(snode, sedege, vnr):
    success = True
    vn = len(vnr.vnode)
    sn = len(snode)
    v2sindex = []
    for vi in range(vn):
        for si in range(sn):
            if vnr.vnode[vi].cpu < snode[si].lastcpu and vnr.id not in [x[0] for x in snode[si].vnodeindexs]:
                snode[si].lastcpu = snode[si].lastcpu - vnr.vnode[vi].cpu
                snode[si].vnodeindexs.append([vnr.id, vnr.vnode[vi].index])
                v2sindex.append(si)
                break
            else:
                if (si == sn - 1):
                    success = False
                    return success, [], [], []
    return success, v2sindex, snode, sedege


def edegemapping(asnode, asedege, vnr, v2sindex):
    success = True
    ve2seindex = []
    ven = len(vnr.vedge)
    for i in range(ven):
        fromnode = vnr.vedge[i].link[0]
        tonode = vnr.vedge[i].link[1]
        fromnode = v2sindex[fromnode]
        tonode = v2sindex[tonode]
        bandlimit = vnr.vedge[i].bandwidth
        g = Sn2_networkxG(asnode, asedege, bandlimit)
        pathindex, cost = shortpath(g, asedege, fromnode, tonode)
        if not pathindex:
            return False, [], [], []
        for j in pathindex:
            asedege[j].lastbandwidth = asedege[j].lastbandwidth - vnr.vedge[i].bandwidth
            asedege[j].vedgeindexs.append([vnr.id, i])
        ve2seindex.append(pathindex)
    return success, ve2seindex, asnode, asedege


# EH_ALG
def edegemapping2(asnode, asedege, vnr, v2sindex):
    v2sindex = np.array(v2sindex)
    success = True
    ve2seindex = []
    ven = len(vnr.vedge)
    for i in range(ven):
        fromnode = vnr.vedge[i].link[0]
        tonode = vnr.vedge[i].link[1]

        fromnode = v2sindex[v2sindex[:, 0] == fromnode][0][1]
        tonode = v2sindex[v2sindex[:, 0] == tonode][0][1]
        bandlimit = vnr.vedge[i].bandwidth
        g = Sn2_networkxG(asnode, asedege, bandlimit)
        pathindex, cost = shortpath(g, asedege, fromnode, tonode)
        if (not pathindex) and (cost is not 0):
            return False, [], [], []
        for j in pathindex:
            asedege[j].lastbandwidth = asedege[j].lastbandwidth - vnr.vedge[i].bandwidth
            asedege[j].vedgeindexs.append([vnr.id, i])
        ve2seindex.append(pathindex)
    return success, ve2seindex, asnode, asedege


def edgemapping3(asnode, asedege, myvnr, v2sindex):
    v2sindex = np.array(v2sindex)
    success = True
    ve2seindex = []
    for vedge in myvnr.vedge:
        fromnode = vedge.link[0]
        tonode = vedge.link[1]
        print(fromnode, v2sindex)
        # print(tonode, v2sindex[v2sindex[:, 0] == tonode])
        fromnode = v2sindex[v2sindex[:, 0] == fromnode][0][1]
        tonode = v2sindex[v2sindex[:, 0] == tonode][0][1]
        bandlimit = vedge.bandwidth
        g = Sn2_networkxG(asnode, asedege, bandlimit)
        pathindex, cost = shortpath(g, asedege, fromnode, tonode)
        if (not pathindex) and (cost is not 0):
            return False, [], [], []
        for j in pathindex:
            asedege[j].lastbandwidth = asedege[j].lastbandwidth - vedge.bandwidth
            asedege[j].vedgeindexs.append([myvnr.id, vedge.index])
        ve2seindex.append(pathindex)
    return success, ve2seindex, asnode, asedege


def Sn2_networkxG(snode, sedege, bandlimit):
    g = nx.Graph()
    n = len(snode)
    for i in range(n):
        g.add_node(i)

    en = len(sedege)
    for i in range(en):
        bandwidth = 0
        if sedege[i].bandwidth > bandlimit:
            bandwidth = sedege[i].bandwidth - bandlimit
            g.add_weighted_edges_from([(sedege[i].link[0], sedege[i].link[1], bandwidth)])
    return g


def shortpath(G, sedege, fromnode, tonode):
    try:
        sedegeindex = []
        path = nx.dijkstra_path(G, fromnode, tonode, weight="none")
        for i in range(len(path) - 1):
            sedegeindex.append(getedege(sedege, path[i], path[i + 1]))
        cost = nx.dijkstra_path_length(G, fromnode, tonode)
    except nx.NetworkXNoPath:
        return [], []

    return sedegeindex, cost,


def getedege(sedege, sourcenode, targetnode):
    for i in range(len(sedege)):
        for j in range(len(sedege[i].link)):
            if (sourcenode == sedege[i].link[0] and targetnode == sedege[i].link[1]) or (
                    sourcenode == sedege[i].link[1] and targetnode == sedege[i].link[0]):
                return sedege[i].index
    return []


# 卷积w网络
def cnn():
    pass


def testMapping(snode, sedege, vnr, g, mg):
    pass



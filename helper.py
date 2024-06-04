import numpy as np

def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def disR2PQ(p, q, r):
    """
    khoảng cách từ r đến pq
    """
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)

def disPoint2Point(p, q):
    """
    Khoảng cách từ điểm đến điểm
    """
    return np.linalg.norm(p - q)

def middlePoint(p, q):
    """
    Khoảng cách từ điểm đến điểm
    """
    return (p + q) / 2

#print(middlePoint(np.array((4, 5, 6)), np.array((7, 8, 9))))

#print(d(np.array((1, 2, 3)), np.array((4, 5, 6)), np.array((7, 8, 9))))

print(sum([
0.027569655014139383, 0.26203010667260546, 0.2142289573903439, 0.2895922292102187, 0.2795753065673985]))
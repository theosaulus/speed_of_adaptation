import numpy as np

def compute_cpdag(dag):
    """
    Compute the CPDAG (essential graph) for the Markov equivalence class (MEC) of a 
    given DAG. An edge i -> j in the DAG is undirected in the CPDAG iff parents of j 
    (excluding i) equal the parents of i:
    
    Input: dag : numpy.ndarray of shape (n, n)
    Returns: cpdag : numpy.ndarray of shape (n, n)
    """
    n = dag.shape[0]
    cpdag = dag.copy().astype(np.int8)
    
    for i in range(n):
        for j in range(n):
            if dag[i, j]:
                pa_i = set(np.where(dag[:, i])[0])
                pa_j = set(np.where(dag[:, j])[0])
                if pa_j - {i} == pa_i:
                    cpdag[i, j] = 1
                    cpdag[j, i] = 1
    return cpdag

if __name__ == '__main__':
    dag = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]], dtype=np.int8)

    cpdag = compute_cpdag(dag)
    print("DAG adjacency matrix:")
    print(dag)
    print("\nCPDAG adjacency matrix:")
    print(cpdag)
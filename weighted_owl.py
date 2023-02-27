from scipy.stats import norm
import numpy as np
import cupy as cp


def mydist(w,X, Y): ## Function to calculate the weighted distance between each points of the dataset
    dists = -2 * cp.dot(w*X, Y.T) + cp.sum(w*(Y**2),    axis=1) + cp.sum(w*(X**2), axis=1)[:, cp.newaxis]
    # if prob :
    dists[cp.where(dists <0)] = 0
    return dists

def owlk(dataset,num_clust,beta,nu,sigma,gamma = None,rev = True, wolf_range = 100) : # Main OWL-k-means function
    numdata = dataset.shape[0]
    dim = dataset.shape[1]
    if gamma == None:   ## Initialisation of the gamma's
        q = 0.1
        gamma_bh = norm.ppf(1-np.linspace(1,dim,dim)*q/(2*dim))
        gamma = cp.array(gamma_bh)

    if rev == True:
        gamma = cp.flip(gamma)



    ## standardization
    dataset = (dataset.T * 1/cp.std(dataset,axis =1)).T

    
    # k-means iteration

    from sklearn.cluster import KMeans
    mu=cp.zeros((num_clust, dim))
    weights = cp.array([1/dim for i in range(dim)]) 
    cost_last = cp.inf
    for _ in range(10):
        kmeans = KMeans(n_clusters=num_clust).fit(cp.asnumpy(dataset))
        asso_matrix = cp.zeros((numdata,num_clust))
        for c in range(num_clust):
            # print(np.where(kmeans.labels_ == c ))
            asso_matrix[np.where(kmeans.labels_ == c )[0],c] = 1

        mu_new = cp.array(kmeans.cluster_centers_)
        a = cp.zeros((dim))
        for clust in range(num_clust) :
            dist_clust = (dataset - mu_new[clust]) ** 2
            a_add = asso_matrix[:,clust] @ dist_clust
            a = a + a_add
        
        cost_new = cp.dot(weights**beta ,a)+ nu * cp.dot(cp.abs(weights) ,a) + cp.sum(sigma*gamma * cp.abs(cp.sort(weights)[::-1]))
        if cost_new < cost_last :
            mu = mu_new
            cost_last = cost_new
    weights = cp.array([1/dim for i in range(dim)])

    
        # weights = cp.array(weights)
        




    
    #loop
    cost_last = cp.inf
    
    weights = cp.array([1/dim for i in range(dim)]) 
    for l in range(100) : ## Solving the problem
        # calculate u
        distmatrix = cp.zeros((numdata,num_clust))
        asso_matrix = cp.zeros((numdata,num_clust))
        distmatrix = mydist(weights**beta ,dataset,mu)
        min_array = cp.argmin(distmatrix,axis= 1)
        for i in range(numdata) : # calculate U (the assocaition matrix)
            asso_matrix[i,min_array[i]] = 1
        # calculate Theta (the centroids)
        for clust in range(num_clust) :
            u_clust =asso_matrix[:,clust]
            mu[clust] = cp.zeros((dim))
            mu_data = dataset[cp.where(u_clust != 0)]
            mu[clust] = cp.mean(mu_data,axis=0)
        
        # claculate W

        # a = D_l
        a = cp.zeros((dim))
        for clust in range(num_clust) :
            dist_clust = (dataset - mu[clust]) ** 2
            a_add = asso_matrix[:,clust] @ dist_clust
            a = a + a_add
        
        #frank wolf loop
        for wolf in range(wolf_range) :
            # count = weights.copy()
            # count[count > 0] =1
            # print(np.sum(count))
            a_w = beta * a * (weights ** (beta-1)) + nu * a
            # print(cp.max(a_w))
            argw = cp.argsort(weights)
            argw = cp.argsort(argw)
            l_sorted = sigma * gamma[argw]

            a_w = a_w + l_sorted

            s = cp.zeros((dim))
            s[cp.argmin(a_w)] = 1
            alpha = 2/(wolf+3)
            weights = weights + (alpha * (s - weights))


        weights[cp.isinf(weights)] = 0
        cost = cp.dot(weights**beta ,a)+ nu * cp.dot(cp.abs(weights) ,a) + cp.sum(sigma*gamma * cp.abs(cp.sort(weights)[::-1]))
        
        if cp.abs((cost - cost_last)/cost_last) <= 0.005: ## Condition for the convergence
            # print(cost - cost_last)
            break
        # print(cost - cost_last)
        cost_last = cost

        computed = cp.argmax(asso_matrix,axis = 1)
        # print(l)
    return cp.asnumpy(computed),cp.asnumpy(weights)

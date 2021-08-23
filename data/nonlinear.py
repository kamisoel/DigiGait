import nolds
import numpy as np
from scipy.stats import chi2_contingency


def max_lyapunov_exp(data, delay=None, emb_dim=3,
                     min_tsep=None, min_neighbors=20, 
                     tau=1, trajectory_len=20,):
    """
    Parameters:
    -----------
    data (array-like of float):
    input data
    emb_dim (int):
    embedding dimension for delay embedding
    lag (float):
    lag for delay embedding
    min_tsep (float):
    minimal temporal separation between two “neighbors” (default: find a suitable value by calculating the mean period of the data)
    tau (float):
    step size between data points in the time series in seconds (normalization scaling factor for exponents)
    min_neighbors (int):
    if lag=None, the search for a suitable lag will be stopped when the number of potential neighbors for a vector drops below min_neighbors
    trajectory_len (int):
    the time (in number of data points) to follow the distance trajectories between two neighboring points
    
    Returns:
    -------
    float:
    an estimate of the largest Lyapunov exponent (a positive exponent is a strong indicator for chaos)
    """
    return nolds.lyap_r(data, emb_dim, delay, min_tsep)


def sample_entropy(data, emb_dim=2, tolerance=None):
    """
    The sample entropy of a time series is defined as the negative natural logarithm of the conditional probability that two sequences similar for emb_dim points remain similar at the next point, excluding self-matches.
    A lower value for the sample entropy therefore corresponds to a higher probability indicating more self-similarity.
    
    Parameters:
    -----------
    data (array-like of float):
    input data
    emb_dim (int):
    the embedding dimension (length of vectors to compare)
    tolerance (float):
    distance threshold for two template vectors to be considered equal (default: 0.2 * std(data) at emb_dim = 2, corrected for dimension effect for other values of emb_dim)
    
    Returns:
    --------
    float:
    the sample entropy of the data (negative logarithm of ratio between similar template vectors of length emb_dim + 1 and emb_dim)
    """
    return nolds.sampen(data, emb_dim, tolerance)


def correlation_dim(data, emb_dim, rvals=None):
    """
    Calculates the correlation dimension with the Grassberger-Procaccia algorithm
    
    Parameters:
    -----------
    data (array-like of float):
    time series of data points
    emb_dim (int):
    embedding dimension
    rvals (iterable of float):
    list of values for to use for r (default: logarithmic_r(0.1 * std, 0.5 * std, 1.03))
    
    Returns:
    --------
    float:
    correlation dimension as slope of the line fitted to log(r) vs log(C(r))
    """
    return nolds.corr_dim(data, emb_dim, rvals)


def find_first_local_min(x, y):
    local_min_x = -1
    local_min_y = 1e9
    for i in range(len(x)):
        if y[i] < local_min_y:
            local_min_y = y[i]
            local_min_x = x[i]
        else:
            return local_min_x, local_min_y
    raise ValueError("No min found!")

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]


def mutual_information(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood", 
                                           correction = False)
    mi = 0.5 * g / c_xy.sum()
    return mi


def estimate_delay(data, bins=10, delay_range=(1,100)):
    datDelayInformation = []
    for i in delay_range:
        delayed_data = data[i:]
        shortend_data = data[:-i]
        mi = mutual_information(shortend_data, delayed_data, bins)
        datDelayInformation = np.append(datDelayInformation, [mi])

    x, _ = find_first_local_min(delay_range, datDelayInformation)
    return x


def plot_MI(data, bins=10, delay_range=(1,100)):
    datDelayInformation = []
    for i in delay_range:
        delayed_data = data[i:]
        shortend_data = tata[:-i]
        mi = mutual_information(shortend_data, delayed_data, bins)
        datDelayInformation = np.append(datDelayInformation, [mi])

    x, y = find_first_local_min(range(1,100), datDelayInformation)

    plt.plot(range(1,100),datDelayInformation)
    plt.scatter(x, y, marker='x', color='r')
    plt.xlabel('delay')
    plt.ylabel('mutual information')
    plt.show()


def false_nearest_neighours(data,delay,embeddingDimension):
    "Calculates the number of false nearest neighbours of embedding dimension"

    from sklearn.neighbors import NearestNeighbors
    embeddedData = takensEmbedding(data,delay,embeddingDimension);
    #the first nearest neighbour is the data point itself, so we choose the second one
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embeddedData.transpose())
    distances, indices = nbrs.kneighbors(embeddedData.transpose())
    #two data points are nearest neighbours if their distance is smaller than the standard deviation
    epsilon = np.std(distances.flatten())
    nFalseNN = 0
    for i in range(0, len(data)-delay*(embeddingDimension+1)):
        if (0 < distances[i,1]) and (distances[i,1] < epsilon) and ( (abs(data[i+embeddingDimension*delay] - data[indices[i,1]+embeddingDimension*delay]) / distances[i,1]) > 10):
            nFalseNN += 1;
    return nFalseNN

def plot_fnn(data, delay, max_dim=7):
    nFNN = []
    for i in range(1,max_dim):
        nFNN.append(false_nearest_neighours(data, delay, i) / len(data))
    plt.plot(range(1,max_dim),nFNN);
    plt.xlabel('embedding dimension');
    plt.ylabel('Fraction of fNN');
    plt.show()


def takensEmbedding (data, delay, dimension):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay*dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')    
    embeddedData = np.array([data[0:len(data)-delay*dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(embeddedData, [data[i*delay:len(data) - delay*(dimension - i)]], axis=0)
    return embeddedData;


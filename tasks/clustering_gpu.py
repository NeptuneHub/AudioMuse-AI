
import logging
from config import GMM_COVARIANCE_TYPE, SPECTRAL_N_NEIGHBORS

logger = logging.getLogger(__name__)

_GPU_AVAILABLE = None
_GPU_CHECK_DONE = False

def _check_cuda_driver_available():
    try:
        import ctypes
        cuda = ctypes.CDLL('libcuda.so.1')
        init_result = cuda.cuInit(0)
        if init_result != 0:
            return False
        device_count = ctypes.c_int()
        result = cuda.cuDeviceGetCount(ctypes.byref(device_count))
        return result == 0 and device_count.value > 0
    except Exception:
        return False

def check_gpu_available():
    global _GPU_AVAILABLE, _GPU_CHECK_DONE

    if _GPU_CHECK_DONE:
        return _GPU_AVAILABLE

    if not _check_cuda_driver_available():
        _GPU_AVAILABLE = False
        _GPU_CHECK_DONE = True
        logger.info("GPU acceleration not available for clustering: CUDA driver not accessible")
        return _GPU_AVAILABLE

    try:
        import cupy as cp
        import cuml  # noqa: F401
        test_array = cp.array([1, 2, 3])
        _ = test_array.sum()
        _GPU_AVAILABLE = True
        logger.info("GPU acceleration is available for clustering (RAPIDS cuML detected)")
    except Exception as e:
        _GPU_AVAILABLE = False
        logger.info(f"GPU acceleration not available for clustering: {e}")

    _GPU_CHECK_DONE = True
    return _GPU_AVAILABLE


def _to_gpu_array(X):
    import cupy as cp
    return X if isinstance(X, cp.ndarray) else cp.asarray(X)


class GPUKMeans:
    def __init__(self, n_clusters, init='k-means++', n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.using_gpu = False

    def fit_predict(self, X):
        if check_gpu_available():
            try:
                from cuml.cluster import KMeans as cuKMeans

                kmeans_kwargs = {
                    'n_clusters': int(self.n_clusters),
                    'init': self.init,
                    'n_init': int(self.n_init),
                    'output_type': 'numpy'
                }
                if self.random_state is not None:
                    kmeans_kwargs['random_state'] = int(self.random_state)

                self.model = cuKMeans(**kmeans_kwargs)

                labels = self.model.fit_predict(_to_gpu_array(X))
                self.cluster_centers_ = self.model.cluster_centers_
                self.labels_ = labels
                self.using_gpu = True

                logger.debug(f"GPU KMeans completed: {self.n_clusters} clusters")
                return labels

            except Exception as e:
                logger.warning(f"GPU KMeans failed, falling back to CPU: {e}")

        from sklearn.cluster import KMeans
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state
        )
        labels = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = labels
        self.using_gpu = False

        logger.debug(f"CPU KMeans completed: {self.n_clusters} clusters")
        return labels


class GPUDBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels_ = None
        self.using_gpu = False

    def fit_predict(self, X):
        if check_gpu_available():
            try:
                from cuml.cluster import DBSCAN as cuDBSCAN

                self.model = cuDBSCAN(
                    eps=self.eps,
                    min_samples=self.min_samples,
                    output_type='numpy'
                )

                labels = self.model.fit_predict(_to_gpu_array(X))
                self.labels_ = labels
                self.using_gpu = True

                logger.debug(f"GPU DBSCAN completed: eps={self.eps}, min_samples={self.min_samples}")
                return labels

            except Exception as e:
                logger.warning(f"GPU DBSCAN failed, falling back to CPU: {e}")

        from sklearn.cluster import DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.model.fit_predict(X)
        self.labels_ = labels
        self.using_gpu = False

        logger.debug(f"CPU DBSCAN completed: eps={self.eps}, min_samples={self.min_samples}")
        return labels


class GPUPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = n_components
        self.using_gpu = False

    def fit_transform(self, X):
        if check_gpu_available():
            try:
                from cuml.decomposition import PCA as cuPCA

                self.model = cuPCA(
                    n_components=self.n_components,
                    output_type='numpy'
                )

                X_transformed = self.model.fit_transform(_to_gpu_array(X))
                self.components_ = self.model.components_
                self.explained_variance_ratio_ = self.model.explained_variance_ratio_
                self.n_components_ = self.model.n_components_
                self.using_gpu = True

                logger.debug(f"GPU PCA completed: {self.n_components_} components")
                return X_transformed

            except Exception as e:
                logger.warning(f"GPU PCA failed, falling back to CPU: {e}")

        from sklearn.decomposition import PCA
        self.model = PCA(n_components=self.n_components)
        X_transformed = self.model.fit_transform(X)
        self.components_ = self.model.components_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        self.n_components_ = self.model.n_components_
        self.using_gpu = False

        logger.debug(f"CPU PCA completed: {self.n_components_} components")
        return X_transformed

    def inverse_transform(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before inverse_transform")

        if self.using_gpu:
            try:
                return self.model.inverse_transform(_to_gpu_array(X))
            except Exception as e:
                logger.warning(f"GPU PCA inverse_transform failed: {e}")

        return self.model.inverse_transform(X)



class GPUGaussianMixture:
    def __init__(self, n_components, covariance_type='full', init_params='k-means++',
                 n_init=10, random_state=None, reg_covar=1e-4):
        from sklearn.mixture import GaussianMixture
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            init_params=init_params,
            n_init=n_init,
            random_state=random_state,
            reg_covar=reg_covar
        )
        self.n_components = n_components
        self.means_ = None
        self.using_gpu = False
        logger.debug("GaussianMixture using CPU (no GPU implementation available)")

    def fit_predict(self, X):
        labels = self.model.fit_predict(X)
        self.means_ = self.model.means_
        return labels


class GPUSpectralClustering:
    def __init__(self, n_clusters, assign_labels='kmeans', affinity='nearest_neighbors',
                 n_neighbors=10, random_state=None, n_init=10, verbose=False):
        from sklearn.cluster import SpectralClustering
        self.model = SpectralClustering(
            n_clusters=n_clusters,
            assign_labels=assign_labels,
            affinity=affinity,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_init=n_init,
            verbose=verbose
        )
        self.n_clusters = n_clusters
        self.using_gpu = False
        logger.debug("SpectralClustering using CPU (no GPU implementation available)")

    def fit_predict(self, X):
        return self.model.fit_predict(X)


def get_clustering_model(method, params, use_gpu=False):
    if not use_gpu:
        from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
        from sklearn.mixture import GaussianMixture

        if method == 'kmeans':
            return KMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10)
        elif method == 'dbscan':
            return DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        elif method == 'gmm':
            return GaussianMixture(
                n_components=params['n_components'],
                covariance_type=GMM_COVARIANCE_TYPE,
                init_params='k-means++',
                n_init=10,
                random_state=None,
                reg_covar=1e-4
            )
        elif method == 'spectral':
            return SpectralClustering(
                n_clusters=params['n_clusters'],
                assign_labels='kmeans',
                affinity='nearest_neighbors',
                n_neighbors=SPECTRAL_N_NEIGHBORS,
                random_state=params.get("random_state"),
                n_init=10,
                verbose=False
            )

    if method == 'kmeans':
        return GPUKMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10)
    elif method == 'dbscan':
        return GPUDBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    elif method == 'gmm':
        return GPUGaussianMixture(
            n_components=params['n_components'],
            covariance_type=GMM_COVARIANCE_TYPE,
            init_params='k-means++',
            n_init=10,
            random_state=None,
            reg_covar=1e-4
        )
    elif method == 'spectral':
        return GPUSpectralClustering(
            n_clusters=params['n_clusters'],
            assign_labels='kmeans',
            affinity='nearest_neighbors',
            n_neighbors=SPECTRAL_N_NEIGHBORS,
            random_state=params.get("random_state"),
            n_init=10,
            verbose=False
        )

    raise ValueError(f"Unsupported clustering method: {method}")


def get_pca_model(n_components, use_gpu=False):
    if not use_gpu:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components)

    return GPUPCA(n_components=n_components)

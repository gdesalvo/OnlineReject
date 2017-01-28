import numpy as np

def unpickle(file):
    """
    file: str
    """
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def read_data(fns):
    """
    fns: list of str
    """
    d = {}
    d['data'] = []
    d['labels'] = []
    if type(fns) == str:
        fns = [fns]
    for fn in fns:
        itr_d = unpickle(fn)
        d['data'].append(itr_d['data'])
        d['labels'].append(itr_d['labels'])
    d['data'] = np.concatenate(d['data'], axis=0)
    d['labels'] = np.concatenate(d['labels'], axis=0)
    return d

def filter_label(data, label):
    """
    data: dict with keys 'data' and 'labels'
    label: int
    """
    fltr_data = []
    for itr in range(len(data['labels'])):
        if data['labels'][itr] == label:
            fltr_data.append(data['data'][itr])
    fltr_data = np.array(fltr_data)
    return fltr_data


def aggregate_data(data, labels):
    """
    data: dict with keys 'data' and 'labels'
    labels: list of int
    """
    agg_data = []
    agg_labels = []
    if type(labels) == int:
        labels = [labels]
    for label in labels:
        fltr_data = filter_label(data, label)
        agg_data.append(fltr_data)
        agg_labels.extend([label] * fltr_data.shape[0])
    agg_data = np.concatenate(agg_data, axis=0)
    agg_labels = np.array(agg_labels)
    return agg_data, agg_labels


def build_pca(data, components):
    """
    data: np.array with shape (n_samples, n_features)
    components: list of int
    """
    if type(components) == int:
        components = [components]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=max(components))
    all_pca_data = pca.fit_transform(data)
    pca_data = np.array([all_pca_data[:,a-1] for a in components]).transpose() 
    return pca_data

def pca_data(labels=[7,8]):
    batches = range(1,6)
    fn_data_list = ["cifar-10-batches-py/data_batch_" + str(n_batch) for n_batch in batches]
    d = read_data(fn_data_list)
    agg_data, agg_labels = aggregate_data(d, labels)
    pca_agg_data = build_pca(agg_data, range(1,6))
    return pca_agg_data, agg_labels

def unpack_meta():
    fn_meta = "batches_meta"
    meta_d = unpickle(fn_meta)
    return meta_d

def write_data(data, labels):
    """
    data: np.array with shape (n_samples, n_components)
    labels: np.array with shape (n_samples,)
    """
    np.savetxt("cifar10pca.txt", data, delimiter=",")
    np.savetxt("cifar10labels.txt", labels, delimiter=",")


if __name__ == "__main__":
    data, labels = pca_data()
    write_data(data, labels)

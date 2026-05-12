from sklearn.decomposition import PCA


def fit_transform_pca(train_X, n_components: int = 1):
    """
    PCA yalnızca train verisi üzerinde fit edilir.

    Automata modeli tek boyutlu veriyle çalışacağı için
    çok değişkenli veriyi PC1'e indirgemekte kullanılır.
    """
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_X)

    return train_pca, pca


def transform_with_pca(X, pca):
    """
    Train üzerinde fit edilmiş PCA ile validation/test verisini dönüştürür.
    """
    return pca.transform(X)
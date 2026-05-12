from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_scaler(method: str = "standard"):
    """
    Normalizasyon nesnesi oluşturur.

    method:
    - "standard": StandardScaler
    - "minmax": MinMaxScaler
    """
    if method == "standard":
        return StandardScaler()

    if method == "minmax":
        return MinMaxScaler()

    raise ValueError(f"Desteklenmeyen normalizasyon yöntemi: {method}")


def fit_transform_scaler(train_X, method: str = "standard"):
    """
    Scaler sadece train verisi üzerinde fit edilir.
    Bu, data leakage oluşmasını engeller.
    """
    scaler = create_scaler(method)
    train_scaled = scaler.fit_transform(train_X)

    return train_scaled, scaler


def transform_with_scaler(X, scaler):
    """
    Train üzerinde fit edilmiş scaler ile validation/test verisini dönüştürür.
    """
    return scaler.transform(X)
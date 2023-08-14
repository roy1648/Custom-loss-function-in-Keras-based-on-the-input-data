def custom_loss(input_ten: Tensor, alpha: float) -> Tensor:
    def loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true = y_true[:, :, 0]
        mask: Tensor = input_ten[:, :, 1]
        y_pred_squ: Tensor = K.squeeze(y_pred, axis=2)
        term1: Tensor = alpha * K.square(mask * (y_true - y_pred_squ))
        term2: Tensor = (1 - alpha) * K.square((1 - mask) * (y_true - y_pred_squ))
        loss: ndarray = term1 + term2
        return K.sum(loss, axis=1)
    return loss
  
def corrpute_rmse(input_ten: Tensor) -> Tensor:
    def corrpute_rmse_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true = y_true[:, :, 0]
        mask: Tensor = input_ten[:, :, 1]
        y_pred_squ: Tensor = K.squeeze(y_pred, axis=2)
        return K.sqrt(K.mean(K.square(mask * (y_true - y_pred_squ))))
    return corrpute_rmse_error
  
def no_corrpute_rmse(input_ten: Tensor) -> Tensor:
    def no_corrpute_rmse_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true = y_true[:, :, 0]
        mask: Tensor = input_ten[:, :, 1]
        y_pred_squ: Tensor = K.squeeze(y_pred, axis=2)
        return K.sqrt(K.mean(K.square((1 - mask) * (y_true - y_pred_squ))))
    return no_corrpute_rmse_error

model.compile(
    loss=custom_loss(inputs, 0.8),
    metrics=[
        corrpute_rmse(inputs),
        no_corrpute_rmse(inputs),
        RootMeanSquaredError(),
    ],
)

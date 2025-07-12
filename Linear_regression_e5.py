# dữ liệu huấn luyện
X = [3, 4, 5, 6]
y = [60, 55, 66, 93]

# tham số huấn luyện
epochs = 10
b = 5
a = 10
lr = 0.001

# Định nghĩa các hàm
def predict(x, a, b):
    return a * x + b
def compute_loss(y_mu, y):
    return (y_mu - y) ** 2
def compute_gradient(x, y_mu, y):
    da = 2 * x * (y_mu - y)
    db = 2 * (y_mu - y)
    return da, db
def update_parameters(a, b, da, db, lr):
    n_a = a - lr * da
    n_b = b - lr * db
    return n_a, n_b

# vòng lập huấn luyện
for epoch in range(epochs):
    for idx in range(len(X)):
        x = X[idx]
        y_mu = predict(x, a, b)

        loss = compute_loss(y_mu, y[idx])
        da, db = compute_gradient(x, y_mu, y[idx])
        a, b = update_parameters(a, b, da, db, lr)

        print(f'Epoch {epoch}, Loss: {loss}, a: {a}, b: {b}')
        # kết thúc vòng huấn luyện

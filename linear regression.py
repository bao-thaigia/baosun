x = 5
y = 66
b = 5
a = 10
lr = 0.01

a = float (a)
b = float (b)
lr = float (lr)

def predict(x, a, b):
    return x *a +b
def compute_loss(y_mu, y):
    return (y_mu - y)**2
def compute_gradient(x, y_mu, y):
    da = 2 * x * (y_mu - y)
    db = 2 * (y_mu - y)
    return da, db

def update_parameters(a, b, da, db, lr):
    n_a = a - lr * da
    n_b = b -lr * db
    return n_a, n_b

y_mu = predict (x, a, b)
print(f'predicted y: {y_mu}')
Loss = compute_loss(y_mu, y)
print(f'Loss: {Loss}')
da, db = compute_gradient(x, y_mu, y)
print(f'Gradient: da = {da}, db = {db}')

n_a, n_b = update_parameters(a, b, da, db, lr)
print(f'Updated weights: a= {n_a}, b = {n_b}')
